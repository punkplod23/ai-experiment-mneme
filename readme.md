# Mneme: Mamba 3 Sovereign in Rust

Mneme is an experimental implementation of a true Selective State Space Model (SSM / Mamba-3 architecture) language model built in pure Rust using HuggingFace's Candle tensor library. Inspired by the philosophical ideas of Gottfried Wilhelm Leibniz on computation, monads, and memory, this project explores building sovereign, linear-time AI systems capable of learning and generating coherent dialogue from classic literature and philosophical texts.

## Architecture Highlights

1. **Selective State Space Mechanism (Mamba SSM)**
   - Replaces conventional quadratic $O(L^2)$ attention with selective state-space duality.
   - Causal 1D depthwise convolution ($K=4$) with SiLU activation.
   - Selective data-dependent projections for $\Delta$ (time delta / softplus), $B$ (input state matrix), and $C$ (output state matrix).
   - Learnable continuous-to-discrete decay matrix $A = -\exp(A_{\log})$, yielding stable exponential state space decay $\bar{A} = \exp(\Delta \cdot A)$.
   - Recurrent hidden state $h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot x_t$ with $O(1)$ constant-time autoregressive inference and linear-time $O(L)$ parallel training.
   - Sovereign multiplicative gating via $\text{SiLU}(z)$.

2. **BPE Subword Tokenizer with Byte-Fallback**
   - 2,048-token Byte-Pair Encoding (BPE) subword vocabulary.
   - Guaranteed 100% byte-fallback coverage: can encode any character, symbol, or language losslessly.
   - Dialogue special tokens: `<system>`, `</system>`, `<user>`, `</user>`, `<assistant>`, `</assistant>`, `<bos>`, `<eos>`.
   - Serialized in human-readable `tokenizer.json`.

3. **Expanded Context & Linear Scaling**
   - 256 subword tokens per chunk ($\sim 1,200+$ characters / $200-250$ English words).
   - Because Mamba scales linearly ($O(L)$), memory and compute do not explode with long sequences.

4. **Comprehensive Dataset Ingestion**
   - Unlike naive sampling that clamps to 16 batches, Mneme partitions the entire dataset into batches and trains comprehensively across 100% of the corpus.
   - Real-time loss tracking, tokens-per-second (TPS) metrics, and safetensors checkpointing.

5. **Optimized Sovereign Sizing**
   - 4 Mamba layers, `D_MODEL` = 256, `d_inner` = 512, `d_state` = 32 ($\sim 2.9$ million parameters).
   - Rapid convergence from scratch, fitting entirely in CPU cache with fast SIMD execution and optional CUDA acceleration.

6. **Conversational Memory & Multi-Turn REPL**
   - Built-in `ConversationManager` preserving multi-turn dialogue history.
   - Budget-aware context rolling: retains system personality prompt and recent dialogue turns within context budget.
   - Interactive REPL console with slash commands (`/history`, `/reset`, `/system`, `/temp`, `/quit`).

## Usage

### 1. Training (Comprehensive Ingestion)

Train the model on text data (e.g. `books/test_facts.txt` or the entire `books/` library):

```bash
cargo run --release -- ingest books/test_facts.txt --epochs 30
```

Train across all books:
```bash
cargo run --release -- ingest books --epochs 5 --batch-size 8
```

To reset existing weights and start from scratch:
```bash
cargo run --release -- ingest books/test_facts.txt --epochs 30 --reset
```

### 2. Conversational Chat (Interactive REPL)

Launch the interactive multi-turn chat console:

```bash
cargo run --release -- chat
```

Inside the chat console:
```text
User > What is calculus?
Mneme > Calculus is the mathematical study of continuous change...
User > Who invented it?
Mneme > Gottfried Wilhelm Leibniz and Isaac Newton developed it independently...
```

Chat commands:
- `/history` : View the active dialogue memory turns
- `/reset`   : Clear conversation history
- `/system <prompt>` : Update the agent's system prompt on the fly
- `/temp <val>`      : Change sampling temperature (e.g. `/temp 0.6`)
- `/exit` or `/quit` : Exit chat session

### 3. One-Shot Completion (CLI / Makefile Compatibility)

To run a single prompt completion directly:

```bash
cargo run --release -- chat "The first principle of nature is" --temp 0.7 --len 100
```

Using `make`:
```bash
make chat S="The nature of mind"
```

### 4. Building

```bash
cargo build --release
```

Or using `make`:
```bash
make build
```

## Running Tests

Run the full test suite (covering BPE tokenizer, conversational memory, Mamba SSM recurrence, step forward, and safetensors persistence):

```bash
cargo test
```
