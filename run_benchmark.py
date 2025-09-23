# run_benchmark.py  (位于 repo 根目录，与 nanoGPT/ 并列)
import argparse, json, os, sys, subprocess
from pathlib import Path


_PINN_PDE_PARAMS = {
    'convection': ['beta', '10'],
    'reaction': ['rho', '5'],
    'wave': ['beta', '10'],
}


def _pinn_device_arg(device_str: str) -> str:
    if device_str is None:
        return "0"
    d = device_str.lower()
    if d == "auto":
        return "0"
    if d == "cuda":
        return "0"
    if d.startswith("cuda:"):
        return d.split(":", 1)[1]
    return d


def main():
    ap = argparse.ArgumentParser("Benchmark nanoGPT trainers")
    ap.add_argument("--ARCH", help="nanoGPT, CNN")
    ap.add_argument("--device", default="cuda", help="auto,cpu,cuda,mps")
    ap.add_argument("--dataset", default="shakespeare_char",
                    help="shakespeare,shakespeare_char,openwebtext-3%,MNIST,openwebtext")
    ap.add_argument("--optimizers", default="MUON_WITH_AUX_ADAM")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=str, default="2")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--betas", type=str, default="0.9,0.999")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--momentum", type=float, default=0.0)
    args = ap.parse_args()

    betas = tuple(float(x) for x in args.betas.split(","))
    seeds = [int(s) for s in args.seeds.split(",")]
    optimizers = [s.strip() for s in args.optimizers.split(",")]

    REPO = Path(__file__).resolve().parent
    workdir = REPO / args.ARCH          # 并列目录

    if args.ARCH == "PINN" and (args.dataset in (None, "", "shakespeare_char")):
        args.dataset = "convection"

    if args.ARCH == "nanoGPT":
        config_py = f"config/train_{args.dataset}.py"
        # 先准备数据
        def _data_ready(dataset):
            d = workdir / "data" / dataset
            return (d / "train.bin").exists() and (d / "val.bin").exists()
        if args.dataset in {"shakespeare", "shakespeare_char", "openwebtext-3%", "MNIST"}:
            if _data_ready(args.dataset):
                print("Data already present. Skipping preparation.")
            else:
                print("Preparing data...")
                prep = [sys.executable, f"data/{args.dataset}/prepare.py"]
                subprocess.run(prep, cwd=workdir, check=True, text=True)
                print("Data preparation completed.")

    for seed in seeds:
        for opt_name in optimizers:
            opt = opt_name.upper()
            # 组装 param_groups
            if args.ARCH == "PINN":
                pinn_dir = REPO / "PINN"
                pde = args.dataset
                if pde in (None, "", "shakespeare_char"):
                    pde = 'convection'
                opt_lower = opt_name.lower()
                cmd = [
                    sys.executable, "run_experiment.py",
                    f"--seed={seed}",
                    f"--pde={pde}",
                    f"--opt={opt_lower}",
                    f"--epochs={args.epochs}",
                    f"--device={_pinn_device_arg(args.device)}",
                ]
                pde_params = _PINN_PDE_PARAMS.get(pde.lower(), None)
                if pde_params:
                    cmd.extend(["--pde_params", *pde_params])
                if opt_lower == 'adam':
                    cmd.extend(["--opt_params", "lr", str(args.lr)])
                env = os.environ.copy()
                log_csv = env.get("LOG_CSV")
                if log_csv:
                    env.setdefault("LOG_CSV", log_csv)
                env.setdefault("WANDB_DISABLED", "true")
                env.setdefault("WANDB_MODE", "disabled")
                print("\n=== Running:", " ".join([str(x) for x in cmd]), "\n", flush=True)
                with subprocess.Popen(
                    cmd, cwd=pinn_dir,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, env=env
                ) as p:
                    for line in p.stdout:
                        print(line, end="")
                    rc = p.wait()
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd)
                continue

            if opt in ("ADAM", "ADAMW"):
                param_groups = [{
                    "group_type":"all", "lr":args.lr,
                    "weight_decay":args.weight_decay, "betas":list(betas), "eps":args.eps
                }]
            elif opt == "SGD":
                param_groups = [{
                    "group_type":"all", "lr":args.lr,
                    "weight_decay":args.weight_decay, "momentum":args.momentum
                }]
            elif opt == "MUON":
                param_groups = [{
                    "group_type":"all", "lr":args.lr,
                    "weight_decay":args.weight_decay, "momentum":args.momentum, "eps":args.eps, "use_muon": True
                }]
            elif opt == "MUON_WITH_AUX_ADAM":
                # 典型做法：矩阵参数走 MUON，其余交给 AdamW
                param_groups = [
                    {"group_type":"hidden", "lr":args.lr, "weight_decay":args.weight_decay,
                     "momentum":args.momentum, "eps":args.eps, "use_muon": True},
                    {"group_type":"other", "lr":args.lr, "weight_decay":args.weight_decay,
                     "betas":list(betas), "eps":args.eps, "use_muon": False},
                ]
            else:
                raise ValueError(f"Unsupported optimizer: {opt}")

            pg_json = json.dumps(param_groups)

            if args.ARCH == "nanoGPT":
                compile_flag = "True" if "cuda" in args.device.lower() else "False"
                cmd = [
                    sys.executable, "-u", "train_merged.py", config_py,
                    f"--optimizer_name={opt}",
                    f"--param_groups={pg_json}",
                    f"--epochs={args.epochs}",
                    f"--batch_size={args.batch_size}",
                    f"--device={args.device}",
                    f"--compile={compile_flag}",
                    f"--seed={seed}",
                ]
            else:
                cmd = [
                    sys.executable, "-u", "airbench94.py", 
                    f"--optimizer_name={opt}",
                    f"--param_groups={pg_json}",
                    f"--epochs={args.epochs}",
                    f"--batch_size={args.batch_size}",
                    f"--device={args.device}",
                ]

            print("\n=== Running:", " ".join(cmd), "\n", flush=True)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            # 流式打印子进程输出
            with subprocess.Popen(
                cmd, cwd=workdir, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            ) as p:
                for line in p.stdout:
                    print(line, end="")
                rc = p.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)

if __name__ == "__main__":
    main()
