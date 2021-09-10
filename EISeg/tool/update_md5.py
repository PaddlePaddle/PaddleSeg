import hashlib
from pathlib import Path

models_dir = Path()
ext = ".pdparams"
for model_path in models_dir.glob("*/*" + ext):
    md5 = hashlib.md5(model_path.read_bytes()).hexdigest()
    md5_path = str(model_path)[: -len(ext)] + ".md5"
    Path(md5_path).write_text(md5)
