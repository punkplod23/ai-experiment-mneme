use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{layer_norm, linear, linear_no_bias, Embedding, LayerNorm, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaConfig {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub dt_rank: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub chunk_size: usize,
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            d_inner: 512,
            d_state: 32,
            d_conv: 4,
            dt_rank: 16,
            num_layers: 4,
            vocab_size: 2048,
            chunk_size: 256,
        }
    }
}

pub struct MambaLayerState {
    pub conv_state: Tensor, // (batch_size, d_conv - 1, d_inner)
    pub ssm_state: Tensor,  // (batch_size, d_inner, d_state)
}

pub fn softplus(x: &Tensor) -> Result<Tensor> {
    let exp_x = x.clamp(-20.0, 20.0)?.exp()?;
    (exp_x + 1.0)?.log()
}

pub struct MambaBlock {
    pub norm: LayerNorm,
    pub in_proj: Linear,
    pub conv_weight: Tensor, // (d_inner, d_conv)
    pub conv_bias: Tensor,   // (d_inner)
    pub x_proj: Linear,      // d_inner -> dt_rank + 2 * d_state
    pub dt_proj: Linear,     // dt_rank -> d_inner
    pub a_log: Tensor,       // (d_inner, d_state)
    pub d: Tensor,           // (d_inner)
    pub out_proj: Linear,    // d_inner -> d_model
    pub cfg: MambaConfig,
}

impl MambaBlock {
    pub fn new(cfg: &MambaConfig, vb: VarBuilder) -> Result<Self> {
        let norm = layer_norm(cfg.d_model, 1e-5, vb.pp("norm"))?;
        let in_proj = linear_no_bias(cfg.d_model, 2 * cfg.d_inner, vb.pp("in_proj"))?;

        let conv_weight = vb.get_with_hints(
            (cfg.d_inner, cfg.d_conv),
            "conv_weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;
        let conv_bias = vb.get_with_hints(
            cfg.d_inner,
            "conv_bias",
            candle_nn::Init::Const(0.0),
        )?;

        let x_proj = linear_no_bias(
            cfg.d_inner,
            cfg.dt_rank + 2 * cfg.d_state,
            vb.pp("x_proj"),
        )?;
        let dt_proj = linear(cfg.dt_rank, cfg.d_inner, vb.pp("dt_proj"))?;

        let a_log = vb.get_with_hints(
            (cfg.d_inner, cfg.d_state),
            "a_log",
            candle_nn::Init::Uniform { lo: 0.5, up: 2.5 },
        )?;
        let d = vb.get_with_hints(cfg.d_inner, "d", candle_nn::Init::Const(1.0))?;

        let out_proj = linear_no_bias(cfg.d_inner, cfg.d_model, vb.pp("out_proj"))?;

        Ok(Self {
            norm,
            in_proj,
            conv_weight,
            conv_bias,
            x_proj,
            dt_proj,
            a_log,
            d,
            out_proj,
            cfg: cfg.clone(),
        })
    }

    /// Full sequence forward pass (used during training and sequence evaluation).
    pub fn forward(&self, u: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = u.dims3()?;

        // 1. Pre-norm
        let u_norm = self.norm.forward(u)?;

        // 2. Input projection into (x, z)
        let xz = self.in_proj.forward(&u_norm)?;
        let x = xz.narrow(D::Minus1, 0, self.cfg.d_inner)?.contiguous()?;
        let z = xz.narrow(D::Minus1, self.cfg.d_inner, self.cfg.d_inner)?.contiguous()?;

        // 3. Causal 1D Depthwise Convolution
        let pad = Tensor::zeros(
            (b_sz, self.cfg.d_conv - 1, self.cfg.d_inner),
            x.dtype(),
            x.device(),
        )?;
        let x_padded = Tensor::cat(&[&pad, &x], 1)?;

        let mut conv_out = self
            .conv_bias
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b_sz, seq_len, self.cfg.d_inner))?;
        for k in 0..self.cfg.d_conv {
            let slice = x_padded.narrow(1, k, seq_len)?;
            let w_k = self
                .conv_weight
                .narrow(1, k, 1)?
                .squeeze(1)?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            conv_out = (conv_out + slice.broadcast_mul(&w_k)?)?;
        }
        let x_conv = candle_nn::ops::silu(&conv_out)?;

        // 4. Selective Projections: dt, B, C
        let x_dbl = self.x_proj.forward(&x_conv)?;
        let dt_raw = x_dbl.narrow(D::Minus1, 0, self.cfg.dt_rank)?.contiguous()?;
        let b_proj = x_dbl.narrow(D::Minus1, self.cfg.dt_rank, self.cfg.d_state)?.contiguous()?;
        let c_proj = x_dbl.narrow(
            D::Minus1,
            self.cfg.dt_rank + self.cfg.d_state,
            self.cfg.d_state,
        )?.contiguous()?;

        let dt = softplus(&self.dt_proj.forward(&dt_raw)?)?;

        // 5. State Space Discretization: A_bar and Bx_bar
        let a = self.a_log.exp()?.neg()?; // (d_inner, d_state)
        let dt_u = dt.unsqueeze(3)?;       // (b_sz, seq_len, d_inner, 1)
        let a_u = a.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, d_inner, d_state)
        let a_bar = dt_u.broadcast_mul(&a_u)?.exp()?; // (b_sz, seq_len, d_inner, d_state)

        let b_u = b_proj.unsqueeze(2)?;   // (b_sz, seq_len, 1, d_state)
        let x_u = x_conv.unsqueeze(3)?;   // (b_sz, seq_len, d_inner, 1)
        let bx = dt_u.broadcast_mul(&x_u.broadcast_mul(&b_u)?)?; // (b_sz, seq_len, d_inner, d_state)

        // 6. SSM Recurrent Scan
        let mut h = Tensor::zeros(
            (b_sz, self.cfg.d_inner, self.cfg.d_state),
            x.dtype(),
            x.device(),
        )?;
        let mut y_steps = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let a_bar_t = a_bar.i((.., t))?;
            let bx_t = bx.i((.., t))?;
            let h_decayed = h.mul(&a_bar_t)?;
            h = (&h_decayed + &bx_t)?;

            let c_t = c_proj.i((.., t))?.unsqueeze(1)?;
            let y_t = h.broadcast_mul(&c_t)?.sum(D::Minus1)?;
            let x_t = x_conv.i((.., t))?;
            let y_skip = (y_t + x_t.broadcast_mul(&self.d)?)?;
            y_steps.push(y_skip);
        }

        let y = Tensor::stack(&y_steps, 1)?; // (b_sz, seq_len, d_inner)

        // 7. Multiplicative Sovereign Gate with z
        let y = y.mul(&candle_nn::ops::silu(&z)?)?;

        // 8. Output projection and residual connection
        let out = self.out_proj.forward(&y)?;
        u + out
    }

    /// Single step forward pass for fast O(1) autoregressive generation with state.
    pub fn step_forward(&self, u_step: &Tensor, state: &mut MambaLayerState) -> Result<Tensor> {
        let (b_sz, _, _) = u_step.dims3()?;

        // 1. Pre-norm
        let u_norm = self.norm.forward(u_step)?;

        // 2. Input projection into (x, z)
        let xz = self.in_proj.forward(&u_norm)?;
        let x = xz.narrow(D::Minus1, 0, self.cfg.d_inner)?.contiguous()?;
        let z = xz.narrow(D::Minus1, self.cfg.d_inner, self.cfg.d_inner)?.contiguous()?;

        // 3. Update conv_state and compute 1D causal conv
        let x_window = Tensor::cat(&[&state.conv_state, &x], 1)?; // (b_sz, d_conv, d_inner)
        state.conv_state = x_window.narrow(1, 1, self.cfg.d_conv - 1)?;

        let mut conv_out = self
            .conv_bias
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b_sz, 1, self.cfg.d_inner))?;
        for k in 0..self.cfg.d_conv {
            let slice = x_window.narrow(1, k, 1)?;
            let w_k = self
                .conv_weight
                .narrow(1, k, 1)?
                .squeeze(1)?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            conv_out = (conv_out + slice.broadcast_mul(&w_k)?)?;
        }
        let x_conv = candle_nn::ops::silu(&conv_out)?;

        // 4. Selective Projections
        let x_dbl = self.x_proj.forward(&x_conv)?;
        let dt_raw = x_dbl.narrow(D::Minus1, 0, self.cfg.dt_rank)?.contiguous()?;
        let b_proj = x_dbl.narrow(D::Minus1, self.cfg.dt_rank, self.cfg.d_state)?.contiguous()?;
        let c_proj = x_dbl.narrow(
            D::Minus1,
            self.cfg.dt_rank + self.cfg.d_state,
            self.cfg.d_state,
        )?.contiguous()?;

        let dt = softplus(&self.dt_proj.forward(&dt_raw)?)?;

        // 5. Discretization
        let a = self.a_log.exp()?.neg()?;
        let dt_u = dt.unsqueeze(3)?;
        let a_u = a.unsqueeze(0)?.unsqueeze(0)?;
        let a_bar = dt_u.broadcast_mul(&a_u)?.exp()?.squeeze(1)?; // (b_sz, d_inner, d_state)

        let b_u = b_proj.unsqueeze(2)?;
        let x_u = x_conv.unsqueeze(3)?;
        let bx = dt_u.broadcast_mul(&x_u.broadcast_mul(&b_u)?)?.squeeze(1)?; // (b_sz, d_inner, d_state)

        // 6. Recurrent State Update
        let prev = state.ssm_state.mul(&a_bar)?;
        state.ssm_state = (&prev + &bx)?;

        let c_u = c_proj.unsqueeze(1)?; // (b_sz, 1, 1, d_state)
        let y_t = state
            .ssm_state
            .unsqueeze(1)?
            .broadcast_mul(&c_u)?
            .sum(D::Minus1)?; // (b_sz, 1, d_inner)
        let y_skip = (y_t + x_conv.broadcast_mul(&self.d)?)?;

        // 7. Multiplicative Gating
        let y = y_skip.mul(&candle_nn::ops::silu(&z)?)?;

        // 8. Output projection and residual
        let out = self.out_proj.forward(&y)?;
        u_step + out
    }
}

pub struct Mamba3Sovereign {
    pub embedding: Embedding,
    pub layers: Vec<MambaBlock>,
    pub ln_f: LayerNorm,
    pub head: Linear,
    pub cfg: MambaConfig,
}

impl Mamba3Sovereign {
    pub fn new(cfg: MambaConfig, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(cfg.vocab_size, cfg.d_model, vb.pp("embedding"))?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        let vb_layers = vb.pp("layers");
        for i in 0..cfg.num_layers {
            layers.push(MambaBlock::new(&cfg, vb_layers.pp(i))?);
        }

        let ln_f = layer_norm(cfg.d_model, 1e-5, vb.pp("ln_f"))?;
        let head = linear(cfg.d_model, cfg.vocab_size, vb.pp("head"))?;

        Ok(Self {
            embedding,
            layers,
            ln_f,
            head,
            cfg,
        })
    }

    /// Full sequence forward pass: returns logits of shape (batch_size, seq_len, vocab_size).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.embedding.forward(x)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        let h = self.ln_f.forward(&h)?;
        self.head.forward(&h)
    }

    /// Initialize empty hidden states for step-by-step autoregressive inference.
    pub fn init_states(&self, batch_size: usize, device: &Device) -> Result<Vec<MambaLayerState>> {
        let mut states = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            let conv_state = Tensor::zeros(
                (batch_size, self.cfg.d_conv - 1, self.cfg.d_inner),
                DType::F32,
                device,
            )?;
            let ssm_state = Tensor::zeros(
                (batch_size, self.cfg.d_inner, self.cfg.d_state),
                DType::F32,
                device,
            )?;
            states.push(MambaLayerState {
                conv_state,
                ssm_state,
            });
        }
        Ok(states)
    }

    /// Single step forward pass: takes (batch_size, 1) and returns (batch_size, 1, vocab_size).
    pub fn forward_step(
        &self,
        x_step: &Tensor,
        states: &mut [MambaLayerState],
    ) -> Result<Tensor> {
        let mut h = self.embedding.forward(x_step)?;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.step_forward(&h, &mut states[i])?;
        }
        let h = self.ln_f.forward(&h)?;
        self.head.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

    #[test]
    fn test_mamba_block_forward_dimensions() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = MambaConfig {
            d_model: 64,
            d_inner: 128,
            d_state: 16,
            d_conv: 4,
            dt_rank: 8,
            num_layers: 2,
            vocab_size: 300,
            chunk_size: 32,
        };

        let block = MambaBlock::new(&cfg, vb.pp("block"))?;
        let input = Tensor::randn(0.0f32, 1.0f32, (2, 16, cfg.d_model), &device)?;
        let output = block.forward(&input)?;

        assert_eq!(output.dims(), &[2, 16, cfg.d_model]);
        Ok(())
    }

    #[test]
    fn test_mamba_model_forward_and_loss() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = MambaConfig {
            d_model: 64,
            d_inner: 128,
            d_state: 16,
            d_conv: 4,
            dt_rank: 8,
            num_layers: 2,
            vocab_size: 300,
            chunk_size: 16,
        };

        let model = Mamba3Sovereign::new(cfg.clone(), vb)?;
        let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW::default())?;

        let input_tokens = Tensor::from_slice(&[10u32, 20, 30, 40, 50, 60, 70, 80], (1, 8), &device)?;
        let target_tokens = Tensor::from_slice(&[20u32, 30, 40, 50, 60, 70, 80, 90], (8,), &device)?;

        let logits = model.forward(&input_tokens)?.reshape((8, cfg.vocab_size))?;
        let loss = candle_nn::loss::cross_entropy(&logits, &target_tokens)?;

        opt.backward_step(&loss)?;
        let loss_val = loss.to_vec0::<f32>()?;
        assert!(loss_val > 0.0);
        Ok(())
    }

    #[test]
    fn test_mamba_step_forward_state() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = MambaConfig {
            d_model: 64,
            d_inner: 128,
            d_state: 16,
            d_conv: 4,
            dt_rank: 8,
            num_layers: 2,
            vocab_size: 300,
            chunk_size: 16,
        };

        let model = Mamba3Sovereign::new(cfg.clone(), vb)?;
        let mut states = model.init_states(1, &device)?;

        let token = Tensor::from_slice(&[15u32], (1, 1), &device)?;
        let logits = model.forward_step(&token, &mut states)?;
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
        Ok(())
    }

    #[test]
    fn test_mamba_save_and_load() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let varmap1 = VarMap::new();
        let vb1 = VarBuilder::from_varmap(&varmap1, DType::F32, &device);
        let cfg = MambaConfig {
            d_model: 64,
            d_inner: 128,
            d_state: 16,
            d_conv: 4,
            dt_rank: 8,
            num_layers: 2,
            vocab_size: 300,
            chunk_size: 16,
        };

        let model1 = Mamba3Sovereign::new(cfg.clone(), vb1)?;
        let test_file = "test_mamba_save.safetensors";
        varmap1.save(test_file)?;

        let mut varmap2 = VarMap::new();
        let vb2 = VarBuilder::from_varmap(&varmap2, DType::F32, &device);
        let model2 = Mamba3Sovereign::new(cfg.clone(), vb2)?;
        varmap2.load(test_file)?;

        let input = Tensor::from_slice(&[5u32, 10, 15], (1, 3), &device)?;
        let out1 = model1.forward(&input)?.to_vec3::<f32>()?;
        let out2 = model2.forward(&input)?.to_vec3::<f32>()?;
        assert_eq!(out1, out2);

        let _ = std::fs::remove_file(test_file);
        Ok(())
    }
}
