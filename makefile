# --- Configuration ---
# Optimized for Windows with AVX-512 / target-cpu=native support
EXE = .\target\release\mamba3_sov.exe
BOOKS_DIR = .\books
# Mamba 3 uses unified safetensors weights
BRAIN_FILE = mamba3_sov.safetensors

# --- Source File Groups ---
L1_FILES = $(wildcard $(BOOKS_DIR)/kjv.txt $(BOOKS_DIR)/shakespeare.txt $(BOOKS_DIR)/aurelius* $(BOOKS_DIR)/plato* $(BOOKS_DIR)/aristotle* $(BOOKS_DIR)/dante* $(BOOKS_DIR)/homer* $(BOOKS_DIR)/milton*)
L2_FILES = $(wildcard $(BOOKS_DIR)/leibniz* $(BOOKS_DIR)/spinoza* $(BOOKS_DIR)/kant* $(BOOKS_DIR)/russell* $(BOOKS_DIR)/descartes* $(BOOKS_DIR)/schopenhauer* $(BOOKS_DIR)/james_psych* $(BOOKS_DIR)/nietzsche* $(BOOKS_DIR)/boole* $(BOOKS_DIR)/mill* $(BOOKS_DIR)/machiavelli* $(BOOKS_DIR)/pearson* $(BOOKS_DIR)/locke* $(BOOKS_DIR)/hume*)
L3_FILES = $(wildcard $(BOOKS_DIR)/faraday* $(BOOKS_DIR)/newton* $(BOOKS_DIR)/darwin* $(BOOKS_DIR)/priestley* $(BOOKS_DIR)/huygens* $(BOOKS_DIR)/curie* $(BOOKS_DIR)/davinci* $(BOOKS_DIR)/steam* $(BOOKS_DIR)/telegraph* $(BOOKS_DIR)/industrial* $(BOOKS_DIR)/tyndall* $(BOOKS_DIR)/hooke*)
L4_FILES = $(wildcard $(BOOKS_DIR)/melville* $(BOOKS_DIR)/austen* $(BOOKS_DIR)/shelley* $(BOOKS_DIR)/dickens* $(BOOKS_DIR)/doyle* $(BOOKS_DIR)/carroll* $(BOOKS_DIR)/thoreau* $(BOOKS_DIR)/joyce* $(BOOKS_DIR)/kafka* $(BOOKS_DIR)/london* $(BOOKS_DIR)/stevenson*)
L5_FILES = $(wildcard $(BOOKS_DIR)/einstein* $(BOOKS_DIR)/turing* $(BOOKS_DIR)/heisenberg*)
L6_FILES = $(wildcard $(BOOKS_DIR)/frost_nixon.txt $(BOOKS_DIR)/galileo_dialogue.txt $(BOOKS_DIR)/berkeley_dialogues.txt $(BOOKS_DIR)/plato_protagoras.txt $(BOOKS_DIR)/hume_dialogues.txt $(BOOKS_DIR)/dryden* $(BOOKS_DIR)/plato_phaedo.txt $(BOOKS_DIR)/cicero*)
L7_FILES = $(wildcard $(BOOKS_DIR)/boole* $(BOOKS_DIR)/mill_logic.txt $(BOOKS_DIR)/pearson* $(BOOKS_DIR)/babbage* $(BOOKS_DIR)/lovelace* $(BOOKS_DIR)/jnevins* $(BOOKS_DIR)/whitehead* $(BOOKS_DIR)/russell*)

# --- Primary Commands ---
.PHONY: all build init clean_brain status chat

# Default: Wipe, Rebuild, and Train the multi-tier hierarchy
all: clean_brain build init

# Compiles with CPU-specific optimizations (AVX-512 / Matrix extensions)
build:
	@echo --- Optimizing for Local CPU Architecture ---
	set RUSTFLAGS=-C target-cpu=native
	cargo build --release

# Sequence: Ingest all books from the library directory
init:
	@echo --- Ingesting Sovereign Library from $(BOOKS_DIR) ---
	$(EXE) ingest $(BOOKS_DIR) --epochs 3
	@echo --- Mamba 3 Sovereign Brain Fully Primed ---

# Deletes the trained weights to start fresh
clean_brain:
	if exist $(BRAIN_FILE) del /f /q $(BRAIN_FILE)

# Quick chat test: usage 'make chat S="Nature of light"'
S ?= "The first principle is"
L ?= 100
chat:
	$(EXE) chat "$(S)" --len $(L)

# GPU-Accelerated Commands (handles MSVC and CUDA paths automatically)
gpu-init:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\run_gpu.ps1 ingest $(BOOKS_DIR) --epochs 3 --batch-size 32

gpu-chat:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\run_gpu.ps1 chat "$(S)" --len $(L)

# Monitoring the health of the Mamba state
status:
	@echo --- Sovereign Brain Metrics ---
	@if exist $(BRAIN_FILE) (powershell -NoProfile "Get-Item $(BRAIN_FILE) | Select-Object Name, Length, LastWriteTime") else (echo Brain file not found.)