import yaml
import os.path as osp
import os

from eiseg import pjpath


def parse_configs(path):
    if not path or not osp.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def save_configs(path=None, config=None, actions=None):
    if not path:
        path = osp.join(pjpath, "config/config.yaml")
    if not osp.exists(path):
        # os.makedirs(osp.basename(path))
        # windows无法使用mknod
        f = open(path, 'w+')
        f.close()
    if not config:
        config = {}
    if actions:
        config["shortcut"] = {}
        for action in actions:
            config["shortcut"][action.data()] = action.shortcut().toString()
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)


class cfgData(object):
    def __init__(self, yaml_file):
        with open(yaml_file, "r", encoding="utf-8") as f:
            fig_data = f.read()
            self.dicts = yaml.load(fig_data)

    def get(self, key):
        if key in self.dicts.keys():
            return self.dicts[key]
        else:
            raise ValueError("Not find this keyword.")


if __name__ == "__main__":
    cfg = cfgData("EISeg/train/train_config.yaml")
    print(cfg.get("use_vdl"))
