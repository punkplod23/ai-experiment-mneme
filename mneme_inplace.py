import os
import re
import unicodedata
from pathlib import Path

TARGET_DIR = "/mnt/c/github/ai-experiment-mneme/original_books"

def sanitize_text(text):
    # 1. Faster Unicode Normalization
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # 2. Convert to lowercase immediately to reduce search space
    text = text.lower()

    # 3. Simple character-by-character filtering (No backtracking risk)
    # We only keep a-z, 0-9, and spaces.
    text = "".join([c if c.isalnum() or c.isspace() else " " for c in text])
    
    # 4. Collapse whitespace with a simple split/join (Much faster than regex)
    return " ".join(text.split())

def format_vault_inplace():
    path_obj = Path(TARGET_DIR)
    if not path_obj.exists():
        print(f"Error: {TARGET_DIR} not found.")
        return

    files = list(path_obj.rglob('*.txt'))
    print(f"Found {len(files)} files. Starting High-Speed Clean...")

    for file_path in files:
        try:
            # We use a smaller read buffer for stability
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content:
                continue

            clean_content = sanitize_text(content)

            # Linux-friendly filename
            new_name = "".join([c if c.isalnum() or c in "._" else "_" for c in file_path.name.lower()])
            new_path = file_path.parent / new_name

            # Write and Rename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(clean_content)

            if file_path != new_path:
                os.replace(file_path, new_path)
                print(f"DONE: {new_name}")
            else:
                print(f"DONE: {file_path.name}")

        except Exception as e:
            print(f"ERROR: {file_path.name} | {e}")

if __name__ == "__main__":
    format_vault_inplace()