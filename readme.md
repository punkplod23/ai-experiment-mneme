# Mneme: Experimental Language Model in Rust

Mneme is an experimental implementation of a transformer-based language model built in Rust. Inspired by the philosophical ideas of Gottfried Wilhelm Leibniz on computation and memory, this project explores building AI systems that can learn and generate text from classic literature.

## Overview

This project implements a complete language model pipeline including:

- **Byte-Pair Encoding (BPE) Tokenizer**: Custom tokenizer trained on the input text corpus
- **Transformer Architecture**: Multi-head self-attention with feed-forward networks
- **Training System**: Adam optimization with gradient clipping and layer normalization
- **Text Generation**: Autoregressive text generation with temperature sampling
- **Memory-Efficient Design**: Uses half-precision (f16) weights for reduced memory footprint

## Architecture

- **Dimensions**: 512 embedding dimension, 8 attention heads
- **Layers**: 6 transformer blocks
- **Context Window**: 16 tokens
- **Vocabulary**: 4000 BPE tokens
- **Training**: Batch size 32, learning rate 0.001

## Usage

### Training (Ingest)

Train the model on text data:

```bash
cargo run --release -- ingest <text_file> --epochs <num_epochs>
```

Example:
```bash
cargo run --release -- ingest books/test.txt --epochs 30
```

### Text Generation (Chat)

Generate text using a trained model:

```bash
cargo run --release -- chat --seed "Once upon a time" --temp 0.8 --len 100
```

### Building

```bash
cargo build --release
```

## Dataset

The model is designed to be trained on text corpora. The `books/` directory contains a collection of public domain classic texts including works by Aristotle, Darwin, Shakespeare, and many others.

## Technical Details

- **Language**: Rust
- **Dependencies**: Uses efficient crates like `rayon` for parallelism, `half` for f16 support, `bincode` for serialization
- **Performance**: Optimized for CPU training with parallel matrix operations
- **Memory**: Quantized weights (f16) to fit larger models in memory

## Goals

This is an educational and experimental project to understand:
- Transformer architecture internals
- Tokenization and text preprocessing
- Optimization algorithms in deep learning
- Memory-efficient AI implementations
- Philosophical connections between computation and cognition

## License

Experimental project - see LICENSE file for details.
