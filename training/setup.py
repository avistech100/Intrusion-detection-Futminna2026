"""
BioAlert — Human/Animal Intrusion Detection System
====================================================
File   : setup.py
Purpose: Google Colab environment setup — run this once at the
         start of every new Colab session before training.

Steps performed
---------------
1. Mount Google Drive so training outputs are persisted.
2. Extract the compressed dataset from Drive into Colab's
   fast local storage (/content/dataset).
3. Verify the class-split distribution so any missing images
   are caught before a long training run begins.

Author : Hassan Adewale Abdulmalik
Project: BioAlert (Final Year Project)
"""

# ---------------------------------------------------------------------------
# 1. Mount Google Drive
# ---------------------------------------------------------------------------
from google.colab import drive

print("Mounting Google Drive...")
drive.mount('/content/drive')
print("Drive mounted successfully.\n")


# ---------------------------------------------------------------------------
# 2. Dataset extraction
# ---------------------------------------------------------------------------
import zipfile
import os

# Paths — update these if you reorganise your Drive folder
ZIP_PATH    = '/content/drive/MyDrive/intrusion_dataset/human_animal_dataset_v2.zip'
EXTRACT_DIR = '/content/dataset'

os.makedirs(EXTRACT_DIR, exist_ok=True)

print(f"Extracting dataset from:\n  {ZIP_PATH}")
print(f"Destination:\n  {EXTRACT_DIR}\n")

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    zf.extractall(EXTRACT_DIR)

print("Extraction complete.\n")


# ---------------------------------------------------------------------------
# 3. Verify dataset structure
#    Expected layout:
#      /content/dataset/human_animal_dataset_v2/
#        train/ {human, animal, other}
#        val/   {human, animal, other}
#        test/  {human, animal, other}
# ---------------------------------------------------------------------------
DATASET_ROOT = os.path.join(EXTRACT_DIR, 'human_animal_dataset_v2')
SPLITS  = ['train', 'val', 'test']
CLASSES = ['human', 'animal', 'other']

print("Dataset distribution:")
print(f"{'Split':<8} {'Class':<10} {'Images':>7}")
print("-" * 28)

total = 0
for split in SPLITS:
    for cls in CLASSES:
        path  = os.path.join(DATASET_ROOT, split, cls)
        count = len(os.listdir(path)) if os.path.exists(path) else 0
        total += count
        status = "" if count > 0 else "  ← MISSING"
        print(f"{split:<8} {cls:<10} {count:>7}{status}")
    print()                          # blank line between splits

print(f"Total images found: {total}")
print("\nSetup complete — ready to run train.py")
