import solaris as sol
import os
config_path = '../yml/sn7_baseline_infer.yml'
config = sol.utils.config.parse(config_path)
print('Config:')
print(config)

# make infernce output dir
os.makedirs(os.path.dirname(config['inference']['output_dir']), exist_ok=True)

inferer = sol.nets.infer.Inferer(config)
inferer()
