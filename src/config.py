"""إعدادات المشروع: عدّل المسارات هنا بس."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# فولدر الصور: لازم يحتوي على japanese_textiles/ و sri_lankan_textiles/
IMAGES_DIR = ROOT / "data" / "images"
METADATA_CSV = ROOT / "data" / "metadata.csv"

INDEX_DIR = ROOT / "index"
EMB_PATH = INDEX_DIR / "embeddings.npy"
META_PATH = INDEX_DIR / "meta.csv"

IMG_SIZE = 224
BATCH_SIZE = 32
TOP_K = 5
SEED = 42

# أسماء عرض للبلدان
COUNTRY_LABELS = {"japan": "اليابان", "sri_lanka": "سريلانكا"}
