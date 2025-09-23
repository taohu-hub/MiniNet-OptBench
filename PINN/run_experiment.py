# external libraries and packages
import os
import argparse
import sys
import traceback
import torch
import re

from src.train_utils import set_random_seed, train
from src.models import PINN

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='initial seed')
    parser.add_argument('--pde', type=str,
                        default='convection', help='PDE type')
    parser.add_argument('--pde_params', nargs='+', type=str,
                        default=None, help='PDE coefficients')
    parser.add_argument('--opt', type=str, default='lbfgs',
                        help='optimizer to use')
    parser.add_argument('--opt_params', nargs='+', type=str,
                        default=None, help='optimizer parameters')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of layers of the neural net')
    parser.add_argument('--num_neurons', type=int, default=50,
                        help='number of neurons per layer')
    parser.add_argument('--loss', type=str, default='mse',
                        help='type of loss function')
    parser.add_argument('--num_x', type=int, default=257,
                        help='number of spatial sample points (power of 2 + 1)')
    parser.add_argument('--num_t', type=int, default=101,
                        help='number of temporal sample points')
    parser.add_argument('--num_res', type=int, default=10000,
                        help='number of sampled residual points')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to run')
    parser.add_argument('--wandb_project', type=str,
                        default='pinns', help='(unused) kept for compatibility')
    parser.add_argument('--device', type=str, default=0, help='GPU to use')

    # Extract arguments from parser
    args = parser.parse_args()
    # set initial seed
    initial_seed = args.seed
    set_random_seed(initial_seed)

    # organize arguments for the experiment into a dictionary for logging purpose
    if torch.cuda.is_available() and str(args.device).lower() != "cpu":
        device_spec = f'cuda:{args.device}'
    else:
        device_spec = 'cpu'

    experiment_args = {
        "initial_seed": args.seed,
        "pde": args.pde,
        "pde_params": args.pde_params,
        "opt": args.opt,
        "opt_params": args.opt_params,
        "num_layers": args.num_layers,
        "num_neurons": args.num_neurons,
        "loss": args.loss,
        "num_x": args.num_x,
        "num_t": args.num_t,
        "num_res": args.num_res, 
        "epochs": args.epochs,
        "device": device_spec
    }

    # print out arguments
    print("Seed set to: {}".format(initial_seed))
    print("Selected PDE type: {}".format(experiment_args["pde"]))
    print("Specified PDE coefficients: {}".format(
        experiment_args["pde_params"]))
    print("Optimizer to use: {}".format(experiment_args["opt"]))
    print("Specified optimizer parameters: {}".format(
        experiment_args["opt_params"]))
    print("Number of layers: {}".format(experiment_args["num_layers"]))
    print("Number of neurons per layer: {}".format(experiment_args["num_neurons"]))
    print("Number of spatial points (x): {}".format(experiment_args["num_x"]))
    print("Number of temporal points (t): {}".format(experiment_args["num_t"]))
    print("Number of random residual points to sample: {}".format(experiment_args["num_res"]))
    print("Number of epochs: {}".format(experiment_args["epochs"]))
    print("GPU to use: {}".format(experiment_args["device"]))

    # initialize model
    model = PINN(in_dim=2, hidden_dim=experiment_args["num_neurons"], out_dim=1,
                 num_layer=experiment_args["num_layers"]).to(experiment_args["device"])
    # train the model
    try:
        metrics, loss_history = train(model,
                        pde_name=experiment_args["pde"],
                        pde_params=experiment_args["pde_params"],
                        loss_name=experiment_args["loss"],
                        opt_name=experiment_args["opt"],
                        opt_params_list=experiment_args["opt_params"],
                        n_x=experiment_args["num_x"],
                        n_t=experiment_args["num_t"],
                        n_res=experiment_args["num_res"],
                        num_epochs=experiment_args["epochs"],
                        device=experiment_args["device"])
    # log error and traceback info, and exit gracefully
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise e

    # Emit a machine-friendly summary so external runners can parse results
    try:
        import json
        print("PINN_RESULT", json.dumps(metrics))
    except Exception:
        pass

    log_csv = os.environ.get("LOG_CSV")
    if log_csv:
        from pathlib import Path
        import csv
        import time
        Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
        if experiment_args["opt_params"]:
            param_groups_repr = " ".join(experiment_args["opt_params"])
        else:
            param_groups_repr = ""

        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "timestamp": timestamp_str,
            "run_name": experiment_args["opt"],
            "optimizer_name": experiment_args["opt"].upper(),
            "seed": initial_seed,
            "dataset": experiment_args["pde"],
            "param_groups": param_groups_repr,
        }
        row.update(metrics)
        file_exists = os.path.exists(log_csv)
        with open(log_csv, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        if loss_history:
            def _sanitize_for_filename(value):
                if value is None:
                    return "none"
                if isinstance(value, (list, tuple)):
                    if len(value) == 0:
                        return "none"
                    return "_".join(_sanitize_for_filename(v) for v in value)
                text = str(value).replace(" ", "_")
                return re.sub(r'[^A-Za-z0-9._-]', '', text)

            loss_dir = Path(log_csv).parent
            timestamp_token = _sanitize_for_filename(timestamp_str.replace(":", "-").replace(" ", "_"))
            opt_token = _sanitize_for_filename(experiment_args["opt"])
            pde_token = _sanitize_for_filename(experiment_args["pde"])
            opt_params_token = _sanitize_for_filename(experiment_args["opt_params"])
            pde_params_token = _sanitize_for_filename(experiment_args["pde_params"])
            loss_filename = f"{timestamp_token}_{opt_token}_{pde_token}_opt-{opt_params_token}_pde-{pde_params_token}.csv"
            loss_path = loss_dir / loss_filename
            with open(loss_path, "w", newline="") as loss_fh:
                loss_writer = csv.writer(loss_fh)
                loss_writer.writerow(["epoch", "loss"])
                for epoch_idx, loss_value in enumerate(loss_history, start=1):
                    loss_writer.writerow([epoch_idx, float(loss_value)])

if __name__ == "__main__":
    main()
