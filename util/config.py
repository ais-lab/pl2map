import yaml
import os

def update_recursive(dict1, dictinfo):
    for k, v in dictinfo.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
def load_config(config_file, default_path=None):
    with open(config_file, 'r') as f:
        cfg_loaded = yaml.load(f, Loader=yaml.Loader)

    base_config_file = cfg_loaded.get('base_config_file')
    if base_config_file is not None:
        cfg = load_config(base_config_file)
    elif (default_path is not None) and (config_file != default_path):
        cfg = load_config(default_path)
    else:
        cfg = dict()
    update_recursive(cfg, cfg_loaded)
    return cfg

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path