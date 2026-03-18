# --- Configuration ---
EXE = .\target\release\leibniz_node.exe
BOOKS_DIR = .\books
VAULT = sovereign_vault

# Source Files
DICT_RAW = .\dictionart\dictionary.txt
DICT_CLEAN = $(BOOKS_DIR)\dictionary_cleaned.txt
KJV = $(BOOKS_DIR)\kjv.txt
PROSE = $(BOOKS_DIR)\prose.txt
CANDLE = $(BOOKS_DIR)\candle.txt

# Define groups based on filename patterns
L1_FILES = $(wildcard $(BOOKS_DIR)/kjv.txt $(BOOKS_DIR)/shakespeare.txt $(BOOKS_DIR)/aurelius* $(BOOKS_DIR)/plato* $(BOOKS_DIR)/aristotle* $(BOOKS_DIR)/dante* $(BOOKS_DIR)/homer* $(BOOKS_DIR)/milton*)
L2_FILES = $(wildcard $(BOOKS_DIR)/leibniz* $(BOOKS_DIR)/spinoza* $(BOOKS_DIR)/kant* $(BOOKS_DIR)/russell* $(BOOKS_DIR)/descartes* $(BOOKS_DIR)/schopenhauer* $(BOOKS_DIR)/james_psych* $(BOOKS_DIR)/nietzsche* $(BOOKS_DIR)/boole* $(BOOKS_DIR)/mill* $(BOOKS_DIR)/machiavelli* $(BOOKS_DIR)/pearson* $(BOOKS_DIR)/locke* $(BOOKS_DIR)/hume*)
L3_FILES = $(wildcard $(BOOKS_DIR)/faraday* $(BOOKS_DIR)/newton* $(BOOKS_DIR)/darwin* $(BOOKS_DIR)/priestley* $(BOOKS_DIR)/huygens* $(BOOKS_DIR)/curie* $(BOOKS_DIR)/davinci* $(BOOKS_DIR)/steam* $(BOOKS_DIR)/telegraph* $(BOOKS_DIR)/industrial* $(BOOKS_DIR)/tyndall* $(BOOKS_DIR)/hooke*)
L4_FILES = $(wildcard $(BOOKS_DIR)/melville* $(BOOKS_DIR)/austen* $(BOOKS_DIR)/shelley* $(BOOKS_DIR)/dickens* $(BOOKS_DIR)/doyle* $(BOOKS_DIR)/carroll* $(BOOKS_DIR)/thoreau* $(BOOKS_DIR)/joyce* $(BOOKS_DIR)/kafka* $(BOOKS_DIR)/london* $(BOOKS_DIR)/stevenson*)
L5_FILES = $(wildcard $(BOOKS_DIR)/einstein* $(BOOKS_DIR)/turing* $(BOOKS_DIR)/heisenberg*)
# Label 6: Dialogues & Transcripts
# Label 6: Dialogues & Transcripts (Interrogative)
L6_FILES = $(wildcard $(BOOKS_DIR)/frost_nixon.txt $(BOOKS_DIR)/galileo_dialogue.txt $(BOOKS_DIR)/berkeley_dialogues.txt $(BOOKS_DIR)/plato_protagoras.txt $(BOOKS_DIR)/hume_dialogues.txt $(BOOKS_DIR)/dryden* $(BOOKS_DIR)/plato_phaedo.txt $(BOOKS_DIR)/cicero*)

# Label 7: Computation & Coding Logic (Foundational CS)
L7_FILES = $(wildcard $(BOOKS_DIR)/boole* $(BOOKS_DIR)/mill_logic.txt $(BOOKS_DIR)/pearson* $(BOOKS_DIR)/babbage* $(BOOKS_DIR)/lovelace* $(BOOKS_DIR)/jnevins* $(BOOKS_DIR)/whitehead* $(BOOKS_DIR)/russell*)
# --- Label to ID Mapping ---
# 0 is usually reserved for _GLOBAL_
ID_KJV = 1
ID_PROSE = 2
ID_CANDLE = 3

# --- Primary Commands ---

.PHONY: all build init clean_db prune status scrub re-vectorize heal

# Default command: Builds the binary and runs the full initialization
all: clean build init

# Compiles the Rust code in high-performance Release mode
clean:
	cargo clean

# Compiles the Rust code in high-performance Release mode
build:
	cargo build --release

# Sequence: Wipe old noise -> Clean Webster -> Anchor Meanings -> Ingest Silos
# Dictionary is handled via 'dictionary' command (Meaning only, no sentence links)
# Prose/KJV are handled via 'ingest' command (Structural links)
init: clean_db
	@echo "--- Step 1: Cleaning Webster's Dictionary ---"
	$(EXE) dictionary $(DICT_RAW)
	@echo "--- Step 3: Ingesting KJV (ID: 2) ---"
	@for %%f in ($(L1_FILES)) do $(EXE) ingest %%f --epochs 2
	@echo "--- Step 3: Ingesting KJV (ID: 2) ---"
	@for %%f in ($(L2_FILES)) do $(EXE) ingest %%f --epochs 2
	@echo "--- Step 4: Ingesting Prose (ID: 3) ---"
	@for %%f in ($(L3_FILES)) do $(EXE) ingest %%f --epochs 2
	@echo "--- Step 5: Ingesting Faraday (ID: 4) ---"
	@for %%f in ($(L4_FILES)) do $(EXE) ingest %%f --epochs 2
	@echo "--- Initialization Complete ---"
	@echo "--- Ingesting Level 5: Modern Theoretical ---"
	@for %%f in ($(L5_FILES)) do $(EXE) ingest %%f --epochs 2
	@echo "--- Ingesting Level 6: Dialogues/Interviews (ID: 6) ---"
	@for %%f in ($(L6_FILES)) do $(EXE) ingest %%f --epochs 2
	@echo "--- Ingesting Level 7: Dialogues/Interviews (ID: 7) ---"
	@for %%f in ($(L7_FILES)) do $(EXE) ingest %%f --epochs 2

# Prune command: usage 'make prune T=2' (defaults to threshold 1)
T ?= 1
prune:
	$(EXE) prune --threshold $(T)

# Removes links pointing to missing vectors (Fixes panics)
scrub:
	$(EXE) scrub

# Reconstructs the 10,000-D manifold from the vocab tree
re-vectorize:
	$(EXE) re-vectorize

# Full maintenance sequence: Repair links and refresh vectors
heal: scrub re-vectorize
	@echo "--- Vault Restored ---"

# Danger: Deletes the database to fix word-salad pollution
clean_db:
	if exist $(VAULT) rmdir /s /q $(VAULT)

# Simple status check of the vault size
status:
	@echo "Database Size:"
	powershell "Get-ChildItem -Recurse -Path $(VAULT) | Measure-Object -Property Length -Sum"

# Useful for debugging the "Clean" logic
view-clean:
	powershell "Get-Content $(DICT_CLEAN) -Head 20"