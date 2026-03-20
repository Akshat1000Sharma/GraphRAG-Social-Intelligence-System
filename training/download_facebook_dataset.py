# Download SNAP Facebook dataset directly

import os
import zipfile

# ─── Download dataset ───
!wget -q https://snap.stanford.edu/data/facebook_large.zip

# ─── Extract to writable directory ───
extract_path = "/kaggle/working/facebook_data"
os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile("facebook_large.zip", 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Dataset ready at:", extract_path)

# ─── Show files ───
print(os.listdir(extract_path))