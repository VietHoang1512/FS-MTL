import sys
import os
import yaml
import logging
import random
import argparse
import json
from tqdm import tqdm
from timeit import default_timer as timer

import numpy as np
import torch
from tensorboardX import SummaryWriter

import losses
import datasets
import metrics
import model_selector
from min_norm_solvers import MinNormSolver, gradient_normalizers
from bypass_bn import enable_running_stats, disable_running_stats
from mtl import PCGrad, CAGrad, IMTL

NUM_EPOCHS = 100


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--method",
    type=str,
    choices=["mgda", "pcgrad", "a-mgda", "a-pcgrad", "cagrad", "a-cagrad", "imtl"],
    help="MTL weight method",
)

parser.add_argument(
    "--lr",
    default=0.0005,
    type=float,
    help="Learning rate.",
)

parser.add_argument(
    "--optimizer",
    default="Adam",
    type=str,
    help="Optimzer.",
)

parser.add_argument(
    "--batch_size",
    default=256,
    type=int,
    help="Training batch size.",
)

parser.add_argument(
    "--adaptive",
    default=True,
    type=str2bool,
    help="True if you want to use the Adaptive SAM.",
)

parser.add_argument("--rho", default=2, type=float, help="Rho parameter for SAM.")

parser.add_argument("--c", type=float, default=0.4, help="c for CAGrad alg.")

parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")

parser.add_argument("--seed", type=int, default=0, help="seed.")

args = parser.parse_args()

args.output_dir = "outputs/SAM/" + str(args).replace(", ", "/").replace(
    "'", ""
).replace("(", "").replace(")", "").replace("Namespace", "")

print("Output directory:", args.output_dir)
os.system("rm -rf " + args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "config.yaml"), "w") as outfile:
    yaml.dump(vars(args), outfile, default_flow_style=False)

logging.basicConfig(
    filename=f"./{args.output_dir}/training.log",
    level=logging.DEBUG,
    filemode="w",
    datefmt="%H:%M:%S",
    format="%(asctime)s :: %(levelname)-8s \n%(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)


def train_multi_task():
    with open("configs.json") as config_params:
        configs = json.load(config_params)

    with open("celeba.json") as json_params:
        params = json.load(json_params)

    exp_identifier = []
    for (key, val) in params.items():
        if "tasks" in key:
            continue
        exp_identifier += ["{}={}".format(key, val)]

    writer = SummaryWriter(log_dir=args.output_dir)

    train_loader, train_dst, val_loader, val_dst = datasets.get_dataset(
        params, configs, args.batch_size, args.num_workers
    )
    loss_fn = losses.get_loss(params)
    metric = metrics.get_metrics(params)

    model = model_selector.get_model(params)
    model_params = []
    for m in model:
        model_params += model[m].parameters()

    if "RMSprop" in args.optimizer:
        optimizer = torch.optim.RMSprop(model_params, lr=args.lr)
    elif "Adam" in args.optimizer:
        optimizer = torch.optim.Adam(model_params, lr=args.lr)
    elif "SGD" in args.optimizer:
        optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9)

    tasks = params["tasks"]
    all_tasks = configs[params["dataset"]]["all_tasks"]
    logging.info("Starting training with parameters \n \t{} \n".format(str(params)))

    n_iter = 0
    loss_init = {}
    best_acc = 0
    for epoch in range(NUM_EPOCHS):
        start = timer()
        logging.info(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        if (epoch + 1) % 10 == 0:
            # Every 50 epoch, half the LR
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.85
            logging.info("Half the learning rate{}".format(n_iter))

        for m in model:
            model[m].train()
        pbar = tqdm(train_loader, total=len(train_loader))
        for batch in pbar:
            n_iter += 1
            # First member is always images
            images = batch[0].cuda()

            labels = {}
            # Read all targets of all tasks
            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels[t] = batch[i + 1].cuda()

            # Scaling the loss functions based on the algorithm choice
            loss_data = {}
            epsilon_weights = {}

            old_weights = dict()
            epsilon_weights = {"rep": dict()}
            old_weights["rep"] = []
            enable_running_stats(model["rep"])
            shared_g = dict()
            for task_id, t in enumerate(tasks):
                # Comptue gradients of each loss function wrt parameters
                optimizer.zero_grad()
                rep, _ = model["rep"](images)
                out_t, _ = model[t](rep)
                loss = loss_fn[t](out_t, labels[t])
                loss_data[t] = loss.item()
                loss.backward()
                epsilon_weights["rep"][t] = []
                task_norms = []

                # CLASSIFIER WEIGHTS
                old_weights[t] = []
                epsilon_weights[t] = []
                for name, param_c in model[t].named_parameters():
                    ew = torch.zeros_like(param_c).data.clone()
                    if param_c.grad is not None:
                        task_norms.append(
                            (
                                (torch.abs(param_c) if args.adaptive else 1.0)
                                * param_c.grad
                            )
                            .norm(p=2)
                            .data.clone()
                        )
                        old_weights[t].append(param_c.data.clone())

                        ew = (
                            (torch.pow(param_c, 2) if args.adaptive else 1.0)
                            * param_c.grad
                        ).data.clone()
                        param_c.grad.zero_()
                    epsilon_weights[t].append(ew)

                task_norm = torch.norm(torch.stack(task_norms), p=2)
                task_scale = (args.rho / (task_norm + 1e-12)).item()
                epsilon_weights[t] = [
                    epsilon_weight * task_scale for epsilon_weight in epsilon_weights[t]
                ]
                model[t].zero_grad()

                # SHARE WEIGHTS
                task_norms = []
                shared_g[t] = []
                for name, param_s in model["rep"].named_parameters():
                    ew = torch.zeros_like(param_s).data.clone()
                    if param_s.grad is not None:
                        task_norms.append(
                            (
                                (torch.abs(param_s) if args.adaptive else 1.0)
                                * param_s.grad
                            )
                            .norm(p=2)
                            .data.clone()
                        )

                        shared_g[t].append(param_s.grad.data.clone())
                        ew = (
                            (torch.pow(param_s, 2) if args.adaptive else 1.0)
                            * param_s.grad
                        ).data.clone()
                        param_s.grad.zero_()
                    epsilon_weights["rep"][t].append(ew)
                    if task_id == 0:
                        old_weights["rep"].append(param_s.data.clone())

                task_norm = torch.norm(torch.stack(task_norms), p=2)
                task_scale = args.rho / (task_norm + 1e-12)
                epsilon_weights["rep"][t] = [
                    epsilon_weight * task_scale
                    for epsilon_weight in epsilon_weights["rep"][t]
                ]
                model["rep"].zero_grad()
            pbar.set_postfix(LOSS=np.mean(list(loss_data.values())))
            del task_norms

            # Stage 2: Compute the SAM gradients
            disable_running_stats(model["rep"])
            SAM_gradient = dict()
            total = len(old_weights["rep"])
            for task_id, t in enumerate(tasks):

                assert (
                    len(epsilon_weights["rep"][t]) == total
                ), f"Encoder weight's length not equal: {len(epsilon_weights['rep'][t])} vs {total}"

                for shared_idx, (name, param_s) in enumerate(
                    model["rep"].named_parameters()
                ):
                    param_s.data = (
                        old_weights["rep"][shared_idx]
                        + epsilon_weights["rep"][t][shared_idx]
                    ).data.clone()
                    if param_s.grad is not None:
                        param_s.grad.zero_()

                for classifier_idx, (name, param_c) in enumerate(
                    model[t].named_parameters()
                ):
                    param_c.data = (
                        old_weights[t][classifier_idx]
                        + epsilon_weights[t][classifier_idx]
                    ).data.clone()
                    if param_c.grad is not None:
                        param_c.grad.zero_()

                rep, _ = model["rep"](images)
                out_t, _ = model[t](rep)
                loss = loss_fn[t](out_t, labels[t])
                loss.backward()
                SAM_gradient[t] = []
                for name, param in model["rep"].named_parameters():
                    if param.grad is not None:
                        SAM_gradient[t].append(param.grad.data.clone())
                        param.grad.zero_()
                    else:
                        SAM_gradient[t].append(torch.zeros_like(param).data.clone())
                for classifier_idx, (name, param) in enumerate(
                    model[t].named_parameters()
                ):
                    param.data = old_weights[t][classifier_idx].data.clone()
                assert (classifier_idx + 1) == len(
                    old_weights[t]
                ), f"Classifier weight's length not equal: {classifier_idx} vs {len(old_weights[t])}"

            if "mgda" in args.method:
                # MGDA:
                if args.method == "a-mgda":
                    with torch.no_grad():
                        sol, _ = MinNormSolver.find_min_norm_element(
                            [SAM_gradient[t] for t in tasks]
                        )

                    for shared_idx, (name, param) in enumerate(
                        model["rep"].named_parameters()
                    ):
                        param.grad = sum(
                            [
                                sol[task_id] * SAM_gradient[t][shared_idx]
                                for task_id, t in enumerate(tasks)
                            ]
                        )
                        param.data = old_weights["rep"][shared_idx].data.clone()
                elif args.method == "mgda":
                    shared_h = dict()
                    for task_id, t in enumerate(tasks):
                        shared_h[t] = []
                        for shared_idx, (name, param) in enumerate(
                            model["rep"].named_parameters()
                        ):
                            shared_h[t].append(
                                SAM_gradient[t][shared_idx] - shared_g[t][shared_idx]
                            )
                    with torch.no_grad():
                        sol_h, _ = MinNormSolver.find_min_norm_element(
                            [shared_h[t] for t in tasks]
                        )
                        sol_g, _ = MinNormSolver.find_min_norm_element(
                            [shared_g[t] for t in tasks]
                        )
                    for shared_idx, (name, param) in enumerate(
                        model["rep"].named_parameters()
                    ):
                        param.grad = sum(
                            [
                                (
                                    sol_h[task_id] * shared_h[t][shared_idx]
                                    + sol_g[task_id] * shared_g[t][shared_idx]
                                )
                                * 40
                                for task_id, t in enumerate(tasks)
                            ]
                        )
                        param.data = old_weights["rep"][shared_idx].data.clone()
            elif "pcgrad" in args.method:
                # PCGrad:
                if args.method == "pcgrad":
                    shared_h = dict()
                    for task_id, t in enumerate(tasks):
                        shared_h[t] = []
                        for shared_idx, (name, param) in enumerate(
                            model["rep"].named_parameters()
                        ):
                            shared_h[t].append(
                                SAM_gradient[t][shared_idx] - shared_g[t][shared_idx]
                            )
                    with torch.no_grad():
                        shared_g = PCGrad([shared_g[t] for t in tasks])
                        shared_h = PCGrad([shared_h[t] for t in tasks])
                    for shared_idx, (name, param) in enumerate(
                        model["rep"].named_parameters()
                    ):
                        param.grad = shared_g[shared_idx] + shared_h[shared_idx]
                        param.data = old_weights["rep"][shared_idx].data.clone()
                elif args.method == "a-pcgrad":
                    with torch.no_grad():
                        SAM_gradient = PCGrad([SAM_gradient[t] for t in tasks])
                    for shared_idx, (name, param) in enumerate(
                        model["rep"].named_parameters()
                    ):
                        param.grad = SAM_gradient[shared_idx]
                        param.data = old_weights["rep"][shared_idx].data.clone()
            elif "cagrad" in args.method:
                # CAGrad:
                if args.method == "cagrad":
                    shared_h = dict()
                    for task_id, t in enumerate(tasks):
                        shared_h[t] = []
                        for shared_idx, (name, param) in enumerate(
                            model["rep"].named_parameters()
                        ):
                            SAM_gradient[t][shared_idx] = SAM_gradient[t][
                                shared_idx
                            ].flatten()
                            shared_g[t][shared_idx] = shared_g[t][shared_idx].flatten()
                            shared_h[t].append(
                                SAM_gradient[t][shared_idx] - shared_g[t][shared_idx]
                            )
                        shared_g[t] = torch.cat(shared_g[t])
                        shared_h[t] = torch.cat(shared_h[t])
                    with torch.no_grad():
                        shared_g = CAGrad(
                            torch.stack([shared_g[t] for t in tasks]), args.c
                        )
                        shared_h = CAGrad(
                            torch.stack([shared_h[t] for t in tasks]), args.c
                        )
                    total_length = 0
                    for shared_idx, (name, param) in enumerate(
                        model["rep"].named_parameters()
                    ):
                        length = param.data.numel()
                        param.grad = (
                            shared_g[total_length : total_length + length]
                            + shared_h[total_length : total_length + length]
                        ).reshape(param.shape)
                        param.data = old_weights["rep"][shared_idx].data.clone()
                        total_length += length
                elif args.method == "a-cagrad":
                    for t in tasks:
                        SAM_gradient[t] = torch.cat(
                            [grad.flatten() for grad in SAM_gradient[t]]
                        )
                    with torch.no_grad():
                        SAM_gradient = CAGrad(
                            torch.stack([SAM_gradient[t] for t in tasks]), args.c
                        )
                    total_length = 0
                    for shared_idx, (name, param) in enumerate(
                        model["rep"].named_parameters()
                    ):
                        length = param.grad.data.numel()
                        param.grad = SAM_gradient[
                            total_length : total_length + length
                        ].reshape(param.shape)
                        param.data = old_weights["rep"][shared_idx].data.clone()
                        total_length += length

            elif args.method == "imtl":
                # IMTL:
                shared_h = dict()
                for task_id, t in enumerate(tasks):
                    shared_h[t] = []
                    for shared_idx, (name, param) in enumerate(
                        model["rep"].named_parameters()
                    ):
                        SAM_gradient[t][shared_idx] = SAM_gradient[t][
                            shared_idx
                        ].flatten()
                        shared_g[t][shared_idx] = shared_g[t][shared_idx].flatten()
                        shared_h[t].append(
                            SAM_gradient[t][shared_idx] - shared_g[t][shared_idx]
                        )
                    shared_g[t] = torch.cat(shared_g[t])
                    shared_h[t] = torch.cat(shared_h[t])
                with torch.no_grad():
                    shared_g = IMTL([shared_g[t] for t in tasks])
                    shared_h = IMTL([shared_h[t] for t in tasks])
                total_length = 0
                for shared_idx, (name, param) in enumerate(
                    model["rep"].named_parameters()
                ):
                    length = param.data.numel()
                    param.grad = (
                        shared_g[total_length : total_length + length]
                        + shared_h[total_length : total_length + length]
                    ).reshape(param.shape)
                    param.data = old_weights["rep"][shared_idx].data.clone()
                    total_length += length

            optimizer.step()

            writer.add_scalar("training_loss", loss.item(), n_iter)
            for t in tasks:
                writer.add_scalar("training_loss_{}".format(t), loss_data[t], n_iter)

        for m in model:
            model[m].zero_grad()
            model[m].eval()

        tot_loss = {}
        tot_loss["all"] = 0.0
        met = {}
        for t in tasks:
            tot_loss[t] = 0.0
            met[t] = 0.0

        num_val_batches = 0
        with torch.no_grad():
            for batch_val in val_loader:
                val_images = batch_val[0].cuda()
                labels_val = {}

                for i, t in enumerate(all_tasks):
                    if t not in tasks:
                        continue
                    labels_val[t] = batch_val[i + 1]
                    labels_val[t] = labels_val[t].cuda()

                val_rep, _ = model["rep"](val_images, None)
                for t in tasks:
                    out_t_val, _ = model[t](val_rep, None)
                    loss_t = loss_fn[t](out_t_val, labels_val[t])
                    tot_loss["all"] += loss_t.item()
                    tot_loss[t] += loss_t.item()
                    metric[t].update(out_t_val, labels_val[t])
                num_val_batches += 1
            accs = []
            for t in tasks:
                writer.add_scalar(
                    "validation_loss_{}".format(t),
                    tot_loss[t] / num_val_batches,
                    n_iter,
                )
                metric_results = metric[t].get_result()
                for metric_key in metric_results:
                    writer.add_scalar(
                        "metric_{}_{}".format(metric_key, t),
                        metric_results[metric_key],
                        n_iter,
                    )
                    accs.append(metric_results[metric_key])
                metric[t].reset()
            writer.add_scalar("validation_loss", tot_loss["all"] / len(val_dst), n_iter)
            acc = np.mean(accs)
            logging.info("Validation accuracy: {} {}".format(acc, accs))
            if acc > best_acc:
                logging.info(
                    f"Validation accuracy improved from {best_acc} to {acc}, Saving model to {args.output_dir}"
                )

                state = {
                    "epoch": epoch + 1,
                    "model_rep": model["rep"].state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                for t in tasks:
                    key_name = "model_{}".format(t)
                    state[key_name] = model[t].state_dict()

                # torch.save(state, f"{args.output_dir}/model.pt")
                best_acc = acc

            end = timer()
            logging.info("Epoch ended in {}s".format(end - start))


if __name__ == "__main__":
    train_multi_task()
