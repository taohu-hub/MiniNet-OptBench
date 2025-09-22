# run_benchmark.py  (位于 repo 根目录，与 nanoGPT/ 并列)
import argparse, json, os, sys, subprocess
from pathlib import Path

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
    ap.add_argument("--learning_rate", type=float, default=1e-3)
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
        for opt in optimizers:
            opt = opt.upper()
            # 组装 param_groups
            if opt in ("ADAM", "ADAMW"):
                param_groups = [{
                    "group_type":"all", "learning_rate":args.learning_rate,
                    "weight_decay":args.weight_decay, "betas":list(betas), "eps":args.eps
                }]
            elif opt == "SGD":
                param_groups = [{
                    "group_type":"all", "learning_rate":args.learning_rate,
                    "weight_decay":args.weight_decay, "momentum":args.momentum
                }]
            elif opt == "MUON":
                param_groups = [{
                    "group_type":"all", "learning_rate":args.learning_rate,
                    "weight_decay":args.weight_decay, "momentum":args.momentum, "eps":args.eps, "use_muon": True
                }]
            elif opt == "MUON_WITH_AUX_ADAM":
                # 典型做法：矩阵参数走 MUON，其余交给 AdamW
                param_groups = [
                    {"group_type":"hidden", "learning_rate":args.learning_rate, "weight_decay":args.weight_decay,
                     "momentum":args.momentum, "eps":args.eps, "use_muon": True},
                    {"group_type":"other", "learning_rate":args.learning_rate, "weight_decay":args.weight_decay,
                     "betas":list(betas), "eps":args.eps, "use_muon": False},
                ]
            else:
                raise ValueError(f"Unsupported optimizer: {opt}")

            pg_json = json.dumps(param_groups)

            if args.ARCH == "nanoGPT":
                cmd = [
                    sys.executable, "-u", "train_merged.py", config_py,
                    f"--optimizer_name={opt}",
                    f"--param_groups={pg_json}",
                    f"--epochs={args.epochs}",
                    f"--batch_size={args.batch_size}",
                    f"--device={args.device}",
                    "--compile=True",              # 建议带上，便于验证
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
