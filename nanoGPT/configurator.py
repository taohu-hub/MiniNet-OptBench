# nanoGPT/configurator.py
"""
Poor Man's Configurator
"""
import sys, os, json
from ast import literal_eval

# 允许新增的 key（不一定都用得上，但安全）
_ALLOWED_NEW_KEYS = {"param_groups", "optimizer", "optimizer_name",
                     "optimizer_class", "epochs", "seed", "max_iters"}

for arg in sys.argv[1:]:
    if '=' not in arg:
        # 说明这是一个配置文件命令，例如: config/train_shakespeare_char.py
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # 形如 --key=value
        assert arg.startswith('--')
        key, val = arg.split('=', 1)
        key = key[2:]

        # 特判：JSON 格式的 param_groups
        if key == "param_groups":
            try:
                attempt = json.loads(val)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for param_groups: {e}")
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
            continue

        # 其它 key：尽量 literal_eval，失败则用原字符串
        try:
            attempt = literal_eval(val)
        except (SyntaxError, ValueError):
            attempt = val

        if key in globals():
            # 类型简单对齐（宽松一些即可）
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            if key in _ALLOWED_NEW_KEYS:
                print(f"Creating new config key: {key} = {attempt}")
                globals()[key] = attempt
            else:
                raise ValueError(f"Unknown config key: {key}")
