"""Build the local image-search index from images under data/images.

Run from the project root: python build_index.py
Images are discovered recursively. The parent folder name becomes the country label
unless a metadata.csv with path,country columns is available.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src import config as C
from src.features import embed_image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def main():
    C.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not C.IMAGES_DIR.exists():
        C.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"No image folder found. Created: {C.IMAGES_DIR}")
        print("Add images in subfolders such as data/images/Egypt/ or data/images/Morocco/, then run again.")
        sys.exit(1)

    paths = sorted(p for p in C.IMAGES_DIR.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        print(f"No images found in {C.IMAGES_DIR}")
        print("Add images to data/images/<country-or-collection>/ and run this script again.")
        sys.exit(1)

    metadata = {}
    if C.METADATA_CSV.exists():
        try:
            raw = pd.read_csv(C.METADATA_CSV)
            path_col = next((c for c in ("path", "image_path", "filename", "file") if c in raw.columns), None)
            country_col = next((c for c in ("country", "country_code", "nation") if c in raw.columns), None)
            if path_col and country_col:
                for _, row in raw.iterrows():
                    metadata[str(row[path_col]).replace("\\", "/")] = str(row[country_col])
        except Exception as exc:
            print(f"Could not read metadata.csv; using folder names instead ({exc})")

    vectors, rows = [], []
    print(f"Found {len(paths)} images. Loading the pretrained ResNet50 model may require an internet connection on first run.")
    for n, path in enumerate(paths, 1):
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                vector = embed_image(image)
            vectors.append(vector)
            rel = path.relative_to(C.ROOT).as_posix()
            country = metadata.get(rel, metadata.get(path.name, path.parent.name))
            rows.append({"path": str(path.resolve()), "country": country, "filename": path.name})
            if n % 25 == 0 or n == len(paths):
                print(f"Processed {n}/{len(paths)} images")
        except (UnidentifiedImageError, OSError) as exc:
            print(f"Skipping unreadable image {path}: {exc}")
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    if not vectors:
        print("No usable images could be indexed.")
        sys.exit(1)
    np.save(C.EMB_PATH, np.asarray(vectors, dtype="float32"))
    pd.DataFrame(rows).to_csv(C.META_PATH, index=False)
    print(f"Index created: {C.EMB_PATH}")
    print(f"Metadata created: {C.META_PATH}")
    print(f"Indexed {len(rows)} images.")

if __name__ == "__main__":
    main()
