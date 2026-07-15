import yaml
from pathlib import Path
from lib.config.defaults import Config
from dataclasses import asdict, is_dataclass

def save_as_yaml(path):
    cfg = Config()
    cfg_dict = asdict(cfg)
    with open(path, 'w') as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, default_flow_style=False)

def load_from_yaml(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Configuration yaml {path} does not exist")
    with open(path, 'r') as f:
        exp_cfg = yaml.safe_load(f) or {}  # dict, 如果是空yaml, safe_load会返回None
    cfg = Config()
    _update_from_dict(cfg, exp_cfg)
    return cfg

def _update_from_dict(cfg_obj, cfg_dict):
    """Given a dataclass object, update attributes from a yaml file."""
    for key, value in cfg_dict.items():
        if not hasattr(cfg_obj, key):
            raise ValueError(f"{key} in yaml does not exist in default config")

        obj_value = getattr(cfg_obj, key)  # cfg_obj当前key的value
        if is_dataclass(obj_value):
            # 如果cfg_obj的当前value还是dataclass, cfg_dict的当前value也必须是个dict
            if not isinstance(value, dict):
                raise ValueError(f"cfg object and yaml have different structure: {key} in yaml does not contain a dict")
            _update_from_dict(obj_value, value)
        else:  # 如果cfg_obj当前value不再是dataclass, 用cfg_dict的value更新cfg_obj的value
            setattr(cfg_obj, key, value)

if __name__ == "__main__":
    save_as_yaml("/experiments/fe108_mastrack.yaml")