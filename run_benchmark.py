#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, inspect, json, os, sys, time
from typing import Dict, Iterable, Optional

import torch, torch.nn as nn
import torch.optim as TOptim

# ==============================
# 1) 自定义优化器（只做显式导入）
# ==============================
CUSTOM_OPT: Dict[str, type] = {}

def _import_custom_opt():
    root = os.path.dirname(__file__)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from opt.DRSOM_reduced_twice import DRSOM_reduced_twice
        CUSTOM_OPT["DRSOM_reduced_twice"] = DRSOM_reduced_twice
    except Exception as e:
        print(f"[warn] import opt.DRSOM_reduced_twice failed: {e}")
    try:
        from opt.DRSOM_reduced import DRSOM_reduced  # 可选
        CUSTOM_OPT["DRSOM_reduced"] = DRSOM_reduced
    except Exception as e:
        print(f"[warn] import opt.DRSOM_reduced failed: {e}")

_import_custom_opt()


# ==============================
# 2) 杂项工具
# ==============================
def needs_closure(opt: torch.optim.Optimizer) -> bool:
    p = inspect.signature(opt.step).parameters.get("closure")
    return p is not None and p.default is inspect._empty

def build_opt(name: str, params: Iterable[nn.Parameter], lr: float, kw: Dict) -> torch.optim.Optimizer:
    if name in CUSTOM_OPT:
        return CUSTOM_OPT[name](params, lr=lr, **kw)
    if hasattr(TOptim, name):
        cls = getattr(TOptim, name)
        try:    return cls(params, lr=lr, **kw)
        except TypeError:
            return cls(params, **kw)
    raise ValueError(f"Unknown optimizer: {name}")

def parse_kv(s: Optional[str]) -> Dict:
    if not s: return {}
    s = s.strip()
    if s.startswith("{"): return json.loads(s)
    out = {}
    for item in s.split(","):
        k, v = item.split("=", 1)
        k = k.strip(); v = v.strip()
        try: out[k] = json.loads(v)
        except Exception:
            if v.lower() in ("true","false"): out[k] = (v.lower()=="true")
            else:
                try: out[k] = float(v) if any(ch in v for ch in ".eE") else int(v)
                except Exception: out[k] = v
    return out

def device_auto(pref: str):
    if pref == "cpu": return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)


# ==============================
# 3) 直接“引用” nanoGPT：模型 + DataLoaderLite
#    —— 不再实现任何 fallback/自定义数据集 ——
# ==============================
def import_nanogpt(nanogpt_path: Optional[str]):
    """
    返回 (model_mod, dataloader_mod)。优先从 --nanogpt-path 导入，
    否则尝试 'nanogpt.*' 包式导入。失败则抛错。
    """
    tried = []
    # from source tree
    if nanogpt_path and os.path.isdir(nanogpt_path):
        if nanogpt_path not in sys.path:
            sys.path.insert(0, nanogpt_path)
        try:
            import model as ng_model
            import dataloader as ng_loader
            return ng_model, ng_loader
        except Exception as e:
            tried.append(f"{nanogpt_path} (as source) -> {e}")

    # as installed/namespace package
    try:
        import nanogpt.model as ng_model  # type: ignore
        import nanogpt.dataloader as ng_loader  # type: ignore
        return ng_model, ng_loader
    except Exception as e:
        tried.append(f"nanogpt.* (as package) -> {e}")

    raise ImportError(
        "无法导入 nanoGPT。请用 --nanogpt-path 指向 karpathy/nanoGPT 源码根目录，"
        "且其中包含 model.py 与 dataloader.py。\nTried:\n  - " + "\n  - ".join(tried)
    )

def build_nanogpt_dataloader(ng_loader, B: int, T: int, data_dir: str, device: torch.device):
    """
    DataLoaderLite 的构造签名在不同版本可能不同。
    这里做一次轻量“自适应”，仍然不自建 Dataset。
    """
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin   = os.path.join(data_dir, "val.bin")
    if not os.path.isfile(train_bin) or not os.path.isfile(val_bin):
        raise FileNotFoundError(f"需要 {train_bin} 与 {val_bin}（请先用 nanoGPT 的 prepare.py 生成）")
    DL = getattr(ng_loader, "DataLoaderLite", None)
    if DL is None:
        raise ImportError("在 nanoGPT.dataloader 中找不到 DataLoaderLite")

    sig = inspect.signature(DL.__init__)
    params = sig.parameters
    # 常见两种：DataLoaderLite(B,T,train_bin,val_bin,device) 或 DataLoaderLite(B,T,data_dir,device)
    if {"train_bin","val_bin"} <= set(params):
        return DL(B, T, train_bin=train_bin, val_bin=val_bin, device=device)
    elif "data_dir" in params:
        return DL(B, T, data_dir=data_dir, device=device)
    else:
        # 最小兜底：按位置参数尝试 (B,T,train,val,device)
        try:
            return DL(B, T, train_bin, val_bin, device)
        except Exception as e:
            raise TypeError(f"无法匹配 DataLoaderLite 构造参数：{e}")

def get_batch_from_loader(dl):
    """
    适配不同版本接口：有的叫 get_batch(split)，有的叫 next_batch()。
    """
    if hasattr(dl, "get_batch"):
        try:
            return dl.get_batch("train")
        except TypeError:
            return dl.get_batch()
    if hasattr(dl, "next_batch"):
        return dl.next_batch()
    raise AttributeError("DataLoaderLite 既无 get_batch 也无 next_batch")


def run_task_nanogpt(task: str, opt_name: str, args, outdir: str, device: torch.device):
    ng_model, ng_loader = import_nanogpt(args.nanogpt_path)

    # 选择数据目录
    if task == "shakespeare":
        data_dir = args.shakespeare_dir or os.path.join(args.nanogpt_path or "", "data", "shakespeare_char")
    else:
        data_dir = args.owt_dir or os.path.join(args.nanogpt_path or "", "data", "openwebtext")
    if not data_dir or not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在：{data_dir} 。请用 --{ 'shakespeare-dir' if task=='shakespeare' else 'owt-dir' } 指定。")

    # DataLoaderLite（由 nanoGPT 提供）
    dl = build_nanogpt_dataloader(ng_loader, args.batch_size, args.block_size, data_dir, device)

    # 模型（由 nanoGPT 提供）
    GPT, GPTConfig = ng_model.GPT, ng_model.GPTConfig
    # vocab_size：DataLoaderLite 在某些版本有属性 vocab_size，否则使用常见默认（GPT-2 bpe=50257）
    vocab_size = getattr(dl, "vocab_size", 50257 if task == "openwebtext" else 65)
    model = GPT(GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout
    )).to(device)

    opt = build_opt(opt_name, model.parameters(), args.lr, parse_kv(args.opt_kwargs))

    subdir = "shakespeare" if task == "shakespeare" else "openwebtext"
    ensure_dir(os.path.join(outdir, subdir))
    csv_path = os.path.join(outdir, subdir, f"{opt_name}.csv")

    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step","loss","wall_time"])
        for step in range(1, (args.steps or 1000) + 1):
            X, Y = get_batch_from_loader(dl)  # 由 nanoGPT 的 DataLoaderLite 提供
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)

            def closure(backward: bool=True):
                opt.zero_grad(set_to_none=True)
                _, loss = model(X, Y)  # nanoGPT 的 forward 支持 (idx, targets) 返回 (logits, loss)
                if backward: loss.backward()
                return loss

            if needs_closure(opt):
                loss = opt.step(closure)
                loss_val = float(loss) if not isinstance(loss, torch.Tensor) else loss.item()
            else:
                loss_val = closure(True).item()
                opt.step()

            w.writerow([step, loss_val, f"{time.time()-t0:.6f}"])

    return csv_path


# ==============================
# 4) 直接“引用” pytorch/examples 的 MNIST: 只复用它的 Net
#    —— 不再自带 SmallMNIST ——
# ==============================
def import_examples_mnist_net(examples_root: Optional[str]):
    """
    从 pytorch/examples 根目录导入 mnist/main.py 并取其中 Net 类。
    """
    if not examples_root or not os.path.isdir(examples_root):
        raise FileNotFoundError("--torch-examples-path 需要指向 pytorch/examples 根目录")
    import importlib.util
    mp = os.path.join(examples_root, "mnist", "main.py")
    if not os.path.isfile(mp):
        raise FileNotFoundError(f"未找到 {mp} （请提供 pytorch/examples 源码路径）")
    spec = importlib.util.spec_from_file_location("mnist_main", mp)
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "Net"):
        raise ImportError("pytorch/examples/mnist/main.py 未导出 Net 类")
    return mod.Net

def run_task_mnist(opt_name: str, args, outdir: str, device: torch.device):
    from torchvision import datasets, transforms
    Net = import_examples_mnist_net(args.torch_examples_path)
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()

    train_ds = datasets.MNIST("./data", train=True, download=True,
                              transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type=="cuda")
    )

    opt = build_opt(opt_name, model.parameters(), args.lr, parse_kv(args.opt_kwargs))

    ensure_dir(os.path.join(outdir, "mnist"))
    csv_path = os.path.join(outdir, "mnist", f"{opt_name}.csv")
    t0 = time.time()
    step = 0
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step","loss","epoch","batch","wall_time"])
        for epoch in range(1, (args.epochs or 1) + 1):
            model.train()
            for b, (x, y) in enumerate(train_loader, start=1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                def closure(backward: bool=True):
                    opt.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits, y)
                    if backward: loss.backward()
                    return loss

                if needs_closure(opt):
                    loss = opt.step(closure)
                    loss_val = float(loss) if not isinstance(loss, torch.Tensor) else loss.item()
                else:
                    loss_val = closure(True).item()
                    opt.step()

                w.writerow([step, loss_val, epoch, b, f"{time.time()-t0:.6f}"])
                step += 1
                if args.steps and step >= args.steps:
                    return csv_path
    return csv_path


# ==============================
# 5) CLI
# ==============================
def main():
    ap = argparse.ArgumentParser("Benchmark optimizers using nanoGPT & pytorch/examples (strict external reuse)")
    ap.add_argument("--tasks", type=str, default="shakespeare", help="shakespeare,openwebtext,mnist（可逗号分隔）")
    ap.add_argument("--optimizers", type=str, default="Adam", help="Adam,SGD,DRSOM_reduced_twice,...（可逗号分隔）")
    ap.add_argument("--opt-kwargs", type=str, default=None, help='优化器参数：JSON 或 key=value，例如 line_search_fn="strong_wolfe",accurate=true')
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    # LM 架构
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    # 外部 repo 路径
    ap.add_argument("--nanogpt-path", type=str, default=None, help="karpathy/nanoGPT 源码根目录（推荐提供）")
    ap.add_argument("--torch-examples-path", type=str, default=None, help="pytorch/examples 根目录（MNIST 需要）")
    # 数据目录（使用 nanoGPT prepare 脚本生成的 bin）
    ap.add_argument("--shakespeare-dir", type=str, default=None, help="含 train.bin/val.bin 的目录（缺省尝试 <nanogpt>/data/shakespeare_char）")
    ap.add_argument("--owt-dir", type=str, default=None, help="含 train.bin/val.bin 的目录（缺省尝试 <nanogpt>/data/openwebtext）")
    args = ap.parse_args()

    device = device_auto(args.device)
    ensure_dir(args.outdir)

    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    opts  = [o.strip() for o in args.optimizers.split(",") if o.strip()]

    print(f"[info] device={device} tasks={tasks} optimizers={opts}")
    for t in tasks:
        if t not in ("shakespeare","openwebtext","mnist"):
            raise ValueError(f"Unknown task: {t}")
        for o in opts:
            print(f"[run] task={t} opt={o}")
            if t == "mnist":
                path = run_task_mnist(o, args, args.outdir, device)
            else:
                path = run_task_nanogpt(t, o, args, args.outdir, device)
            print(f"[done] curve -> {path}")

if __name__ == "__main__":
    main()
