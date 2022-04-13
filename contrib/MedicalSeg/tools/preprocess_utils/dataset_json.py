import json


def parse_msd_basic_info(json_path):
    """
    get dataset basic info from msd dataset.json
    """
    dict = json.loads(open(json_path, "r").read())
    info = {}
    info["modalities"] = tuple(dict["modality"].values())
    info["labels"] = dict["labels"]
    info["dataset_name"] = dict["name"]
    info["dataset_description"] = dict["description"]
    info["license_desc"] = dict["licence"]
    info["dataset_reference"] = dict["reference"]
    return info
