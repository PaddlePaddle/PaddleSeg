from pathlib import Path, PurePath
import wget
import hashlib

# url = "http://182.61.31.110/model/hrnet18s_ocr48_human_f_007.pdparams"
# url = "http://localhost/hrnet18s_ocr48_human_f_007.pdparams"


def model_path(name, refresh=False):
    local_path = Path.home() / Path(".EISeg/model", name, name + ".pdparams")
    print(local_path)

    if local_path.exists():
        return str(local_path)

    def bar_custom(current, total, width=80):
        print(current, total)

    def bar_dummy(current, total, width=80):
        pass

    for f in local_path.parent.glob("*.tmp"):
        f.unlink()
    if not local_path.parent.exists():
        local_path.parent.mkdir()

    # base_url = "http://182.61.31.110"
    base_url = "http://localhost"
    param_url = f"{base_url}/model/{name}/{name}.pdparams"
    md5_url = f"{base_url}/model/{name}/{name}.md5"
    md5_path = local_path.parent / Path(name + ".md5")
    wget.download(md5_url, str(md5_path), bar_dummy)
    remote_md5 = md5_path.read_text()

    for _ in range(5):
        wget.download(param_url, str(local_path), bar_custom)
        local_md5 = hashlib.md5(local_path.read_bytes()).hexdigest()
        if local_md5 == remote_md5:
            md5_path.unlink()
            return str(local_path)
    print("error")
    return None


# model_path("hrnet18s_ocr48_human_f_007")
