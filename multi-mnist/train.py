import os
import sys

import argparse
import logging
import pickle
import yaml

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_lenet import LenetModel
from model_resnet import ResnetModel
from utils import setup_seed, MinNormSolver
from bypass_bn import enable_running_stats, disable_running_stats
from mtl import PCGrad, CAGrad, IMTL


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dset",
    default="multi_fashion_and_mnist",
    type=str,
    help="Dataset for training.",
)

parser.add_argument(
    "--method",
    type=str,
    choices=[
        "mgda",
        "pcgrad",
        "a-mgda",
        "a-pcgrad",
        "cagrad",
        "a-cagrad",
        "imtl",
        "ew",
    ],
    help="MTL weight method",
)

parser.add_argument(
    "--batch_size",
    default=256,
    type=int,
    help="Batch size.",
)

parser.add_argument(
    "--lr", default=1e-3, type=float, help="The initial learning rate for SGD."
)

parser.add_argument(
    "--n_epochs",
    default=100,
    type=int,
    help="Total number of training epochs to perform.",
)

parser.add_argument("--c", type=float, default=0.4, help="c for CAGrad alg.")

parser.add_argument(
    "--adaptive",
    default=False,
    type=str2bool,
    help="True if you want to use the Adaptive SAM.",
)

parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")

parser.add_argument(
    "--rho_eval", nargs="*", help="Rho parameter for evaluating sharpness."
)

parser.add_argument(
    "--adaptive_eval",
    default=False,
    type=str2bool,
    help="True if you want to use the Adaptive SAM evaluation.",
)

parser.add_argument("--seed", type=int, default=0, help="seed")

args = parser.parse_args()

args.output_dir = "outputs/" + str(args).replace(", ", "/").replace("'", "").replace(
    "(", ""
).replace(")", "").replace("Namespace", "")

print("Output directory:", args.output_dir)
os.system("rm -rf " + args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "config.yaml"), "w") as outfile:
    yaml.dump(vars(args), outfile, default_flow_style=False)

log_file = os.path.join(args.output_dir, "MOO-SAM.log")

logging.basicConfig(
    filename=f"./{args.output_dir}/{args.dset}.log",
    level=logging.DEBUG,
    filemode="w",
    datefmt="%H:%M:%S",
    format="%(asctime)s :: %(levelname)-8s \n%(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

setup_seed(args.seed)

with open(f"./data/{args.dset}.pickle", "rb") as f:
    trainX, trainLabel, testX, testLabel = pickle.load(f)
trainX = torch.from_numpy(trainX.reshape(120000, 1, 36, 36)).float()
trainLabel = torch.from_numpy(trainLabel).long()
testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
testLabel = torch.from_numpy(testLabel).long()
train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
test_set = torch.utils.data.TensorDataset(testX, testLabel)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=args.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=args.batch_size, shuffle=False
)
logging.info("==>>> total trainning batch number: {}".format(len(train_loader)))
logging.info("==>>> total testing batch number: {}".format(len(test_loader)))


criterion = nn.CrossEntropyLoss()
model = ResnetModel(2).cuda()

param_amount = 0
for p in model.named_parameters():
    param_amount += p[1].numel()
    print(p[0], p[1].numel())
logging.info(f"total param amount: {param_amount}")

shared_optimizer = torch.optim.Adam(
    model.get_shared_parameters(),
    lr=args.lr,  # momentum=0.9
)

classifier_optimizer = torch.optim.Adam(
    model.get_classifier_parameters(),
    lr=args.lr,  # momentum=0.9
)


def train(epoch):
    all_losses_1 = 0
    all_losses_2 = 0
    for (it, batch) in tqdm(
        enumerate(train_loader),
        desc=f"Training on epoch [{epoch}/{args.n_epochs}]",
        total=len(train_loader),
    ):

        X = batch[0]
        y = batch[1]
        X, y = X.cuda(), y.cuda()
        batchsize_cur = X.shape[0]

        ##### Eval #####
        model.train()
        model.zero_grad()

        for rho in args.rho_eval:
            enable_running_stats(model)
            out1, out2 = model(X)

            ##### SAM stage 1, task 1 #####
            loss1 = criterion(out1, y[:, 0])

            loss1.backward(retain_graph=True)
            natural_loss1 = loss1.item()
            task1_norms = []
            task1_ew = []
            task1_old_w = []
            task1_g = []
            old_w = []
            for name, param in model.named_parameters():
                old_w.append(param.data.clone())

                if "task" not in name:
                    task1_g.append(param.grad.clone())
                if "task_2" not in name:
                    task1_norms.append(
                        (
                            (
                                (torch.abs(param) if args.adaptive_eval else 1.0)
                                * param.grad
                            )
                            .norm(p=2)
                            .data.clone()
                        )
                    )
                    ew = (
                        torch.pow(param, 2) if args.adaptive_eval else 1.0
                    ) * param.grad
                    task1_ew.append(ew.flatten())
                    task1_old_w.append(param.data.clone().flatten())
                    param.grad.zero_()

            task1_norm = torch.norm(torch.stack(task1_norms), p=2)
            task1_scale = rho / (task1_norm + 1e-12)

            task1_ew = torch.cat(task1_ew, dim=0) * task1_scale
            task1_old_w = torch.cat(task1_old_w, dim=0)
            ##### SAM stage 1, task 2 #####
            loss2 = criterion(out2, y[:, 1])
            natural_loss2 = loss2.item()
            loss2.backward()
            task2_g = []
            task2_norms = []
            task2_ew = []
            task2_old_w = []
            for name, param in model.named_parameters():
                if "task" not in name:
                    task2_g.append(param.grad.clone())
                if "task_1" not in name:
                    task2_norms.append(
                        (
                            (
                                (torch.abs(param) if args.adaptive_eval else 1.0)
                                * param.grad
                            )
                            .norm(p=2)
                            .data.clone()
                        )
                    )
                    ew = (
                        torch.pow(param, 2) if args.adaptive_eval else 1.0
                    ) * param.grad
                    task2_ew.append(ew.flatten())
                    task2_old_w.append(param.data.clone().flatten())
                    param.grad.zero_()

            task2_norm = torch.norm(torch.stack(task2_norms), p=2)
            task2_scale = rho / (task2_norm + 1e-12)

            task2_ew = torch.cat(task2_ew, dim=0) * task2_scale
            task2_old_w = torch.cat(task2_old_w, dim=0)

            all_losses_1 += loss1.detach().cpu().numpy() * batchsize_cur
            all_losses_2 += loss2.detach().cpu().numpy() * batchsize_cur

            ##### SAM stage 2, task 1 #####
            task1_new_w = (task1_old_w + task1_ew).data.clone()

            task1_index = 0
            for name, param in model.named_parameters():
                if "task_2" in name:
                    continue
                length = param.flatten().shape[0]
                param.data = task1_new_w[task1_index : task1_index + length].reshape(
                    param.shape
                )
                task1_index += length

            assert task1_index == len(
                task1_new_w
            ), f"Redundant param: {task1_index} vs {len(task1_new_w)}"

            model.zero_grad()

            out1, _ = model(X)
            loss1 = criterion(out1, y[:, 0])
            perturbed_loss_1 = loss1.item()
            loss1.backward()
            task1_classifier_grad = []
            task1_shared_grad = []
            for name, param in model.named_parameters():
                if "task" in name:
                    if "task_1" in name:
                        task1_classifier_grad.append(param.grad.data.clone())
                else:
                    task1_shared_grad.append(param.grad.detach().data.clone())

                param.grad.zero_()

            ##### SAM stage 2, task 2 #####
            task2_new_w = (task2_old_w + task2_ew).data.clone()
            task2_index = 0

            for name, param in model.named_parameters():
                if "task_1" in name:
                    continue
                length = param.flatten().shape[0]
                param.data = task2_new_w[task2_index : task2_index + length].reshape(
                    param.shape
                )
                task2_index += length

            assert task2_index == len(
                task2_new_w
            ), f"Redundant param: {task2_index} vs {len(task2_new_w)}"

            model.zero_grad()

            # disable_running_stats(model)
            _, out2 = model(X)
            loss2 = criterion(out2, y[:, 1])
            perturbed_loss_2 = loss2.item()
            loss2.backward()
            task2_classifier_grad = []
            task2_shared_grad = []
            for name, param in model.named_parameters():
                if "task" in name:
                    if "task_2" in name:
                        task2_classifier_grad.append(param.grad.data.clone())
                else:
                    task2_shared_grad.append(param.grad.detach().data.clone())

                param.grad.zero_()

            assert (
                len(task1_shared_grad)
                == len(task2_shared_grad)
                == len(task1_g)
                == len(task2_g)
            ), f"Length mismatch: {len(task1_shared_grad)} vs {len(task2_shared_grad)} vs {len(task1_g)} vs {len(task2_g)}"

            total = len(task1_shared_grad)
            task1_h = [task1_shared_grad[i] - task1_g[i] for i in range(total)]
            task2_h = [task2_shared_grad[i] - task2_g[i] for i in range(total)]

            task1_h_norm = (
                torch.cat([grad.flatten() for grad in task1_h], dim=0).norm(2).item()
            )
            task2_h_norm = (
                torch.cat([grad.flatten() for grad in task2_h], dim=0).norm(2).item()
            )
            surrogate_loss_1[rho].append(perturbed_loss_1 - natural_loss1)
            surrogate_loss_2[rho].append(perturbed_loss_2 - natural_loss2)
            grad_norm_1[rho].append(task1_h_norm)
            grad_norm_2[rho].append(task2_h_norm)

            index_w = 0

            for name, param in model.named_parameters():
                param.data = old_w[index_w]
                index_w += 1
            model.zero_grad()

        ### Train ###
        model.train()
        model.zero_grad()

        enable_running_stats(model)
        out1, out2 = model(X)

        ##### SAM stage 1, task 1 #####
        loss1 = criterion(out1, y[:, 0])

        loss1.backward(retain_graph=True)
        task1_norms = []
        task1_ew = []
        task1_old_w = []
        task1_g = []
        old_w = []
        for name, param in model.named_parameters():
            old_w.append(param.data.clone())

            if "task" not in name:
                task1_g.append(param.grad.clone())
            if "task_2" not in name:
                task1_norms.append(
                    (
                        ((torch.abs(param) if args.adaptive else 1.0) * param.grad)
                        .norm(p=2)
                        .data.clone()
                    )
                )
                ew = (torch.pow(param, 2) if args.adaptive else 1.0) * param.grad
                task1_ew.append(ew.flatten())
                task1_old_w.append(param.data.clone().flatten())
                param.grad.zero_()

        task1_norm = torch.norm(torch.stack(task1_norms), p=2)
        task1_scale = args.rho / (task1_norm + 1e-12)

        task1_ew = torch.cat(task1_ew, dim=0) * task1_scale
        task1_old_w = torch.cat(task1_old_w, dim=0)
        ##### SAM stage 1, task 2 #####
        loss2 = criterion(out2, y[:, 1])
        loss2.backward()
        task2_g = []
        task2_norms = []
        task2_ew = []
        task2_old_w = []
        for name, param in model.named_parameters():
            if "task" not in name:
                task2_g.append(param.grad.clone())
            if "task_1" not in name:
                task2_norms.append(
                    (
                        ((torch.abs(param) if args.adaptive else 1.0) * param.grad)
                        .norm(p=2)
                        .data.clone()
                    )
                )
                ew = (torch.pow(param, 2) if args.adaptive else 1.0) * param.grad
                task2_ew.append(ew.flatten())
                task2_old_w.append(param.data.clone().flatten())
                param.grad.zero_()

        task2_norm = torch.norm(torch.stack(task2_norms), p=2)
        task2_scale = args.rho / (task2_norm + 1e-12)

        task2_ew = torch.cat(task2_ew, dim=0) * task2_scale
        task2_old_w = torch.cat(task2_old_w, dim=0)

        all_losses_1 += loss1.detach().cpu().numpy() * batchsize_cur
        all_losses_2 += loss2.detach().cpu().numpy() * batchsize_cur

        ##### SAM stage 2, task 1 #####
        task1_new_w = (task1_old_w + task1_ew).data.clone()

        task1_index = 0
        for name, param in model.named_parameters():
            if "task_2" in name:
                continue
            length = param.flatten().shape[0]
            param.data = task1_new_w[task1_index : task1_index + length].reshape(
                param.shape
            )
            task1_index += length

        assert task1_index == len(
            task1_new_w
        ), f"Redundant param: {task1_index} vs {len(task1_new_w)}"

        model.zero_grad()

        disable_running_stats(model)
        out1, _ = model(X)
        loss1 = criterion(out1, y[:, 0])
        loss1.backward()
        task1_classifier_grad = []
        task1_shared_grad = []
        for name, param in model.named_parameters():
            if "task" in name:
                if "task_1" in name:
                    task1_classifier_grad.append(param.grad.data.clone())
            else:
                task1_shared_grad.append(param.grad.detach().data.clone())

            param.grad.zero_()

        ##### SAM stage 2, task 2 #####
        task2_new_w = (task2_old_w + task2_ew).data.clone()
        task2_index = 0

        for name, param in model.named_parameters():
            if "task_1" in name:
                continue
            length = param.flatten().shape[0]
            param.data = task2_new_w[task2_index : task2_index + length].reshape(
                param.shape
            )
            task2_index += length

        assert task2_index == len(
            task2_new_w
        ), f"Redundant param: {task2_index} vs {len(task2_new_w)}"

        model.zero_grad()

        # disable_running_stats(model)
        _, out2 = model(X)
        loss2 = criterion(out2, y[:, 1])
        loss2.backward()
        task2_classifier_grad = []
        task2_shared_grad = []
        for name, param in model.named_parameters():
            if "task" in name:
                if "task_2" in name:
                    task2_classifier_grad.append(param.grad.data.clone())
            else:
                task2_shared_grad.append(param.grad.detach().data.clone())

            param.grad.zero_()

        assert (
            len(task1_shared_grad)
            == len(task2_shared_grad)
            == len(task1_g)
            == len(task2_g)
        ), f"Length mismatch: {len(task1_shared_grad)} vs {len(task2_shared_grad)} vs {len(task1_g)} vs {len(task2_g)}"

        total = len(task1_shared_grad)
        task1_h = [task1_shared_grad[i] - task1_g[i] for i in range(total)]
        task2_h = [task2_shared_grad[i] - task2_g[i] for i in range(total)]

        with torch.no_grad():
            if args.method == "ew":
                shared_grad = [
                    (task1_shared_grad[i] + task2_shared_grad[i]) / 2
                    for i in range(total)
                ]
            elif "mgda" in args.method:
                # MGDA:
                if args.method == "mgda":
                    task1_h = torch.cat([grad.flatten() for grad in task1_h], dim=0)
                    task2_h = torch.cat([grad.flatten() for grad in task2_h], dim=0)
                    sol_h, _ = MinNormSolver.find_min_norm_element([task1_h, task2_h])
                    shared_h = sol_h[0] * task1_h + sol_h[1] * task2_h

                    task1_g = torch.cat([grad.flatten() for grad in task1_g], dim=0)
                    task2_g = torch.cat([grad.flatten() for grad in task2_g], dim=0)
                    sol_g, _ = MinNormSolver.find_min_norm_element([task1_g, task2_g])
                    shared_g = sol_g[0] * task1_g + sol_g[1] * task2_g
                    shared_grad_flatten = shared_g + shared_h

                elif args.method == "a-mgda":
                    task1_shared_grad = torch.cat(
                        [grad.flatten() for grad in task1_shared_grad], dim=0
                    )
                    task2_shared_grad = torch.cat(
                        [grad.flatten() for grad in task2_shared_grad], dim=0
                    )
                    sol, _ = MinNormSolver.find_min_norm_element(
                        [task1_shared_grad, task2_shared_grad]
                    )
                    shared_grad_flatten = (
                        sol[0] * task1_shared_grad + sol[1] * task2_shared_grad
                    )

                shared_grad_id = 0
                shared_grad = []
                for name, param in model.named_parameters():
                    if "task" not in name:
                        length = param.grad.flatten().shape[0]
                        shared_grad.append(
                            shared_grad_flatten[
                                shared_grad_id : shared_grad_id + length
                            ].reshape(param.shape)
                        )
                        shared_grad_id += length

            elif "pcgrad" in args.method:
                # PCGrad:
                if args.method == "pcgrad":
                    share_h = PCGrad([task1_h, task2_h])
                    share_g = PCGrad([task1_g, task2_g])
                    shared_grad = [share_g[i] + share_h[i] for i in range(total)]
                elif args.method == "a-pcgrad":
                    shared_grad = PCGrad([task1_shared_grad, task2_shared_grad])
            elif "cagrad" in args.method:
                # CAGrad:
                if args.method == "cagrad":
                    task1_h = torch.cat([grad.flatten() for grad in task1_h], dim=0)
                    task2_h = torch.cat([grad.flatten() for grad in task2_h], dim=0)
                    shared_h = CAGrad(torch.stack([task1_h, task2_h]), args.c)

                    task1_g = torch.cat([grad.flatten() for grad in task1_g], dim=0)
                    task2_g = torch.cat([grad.flatten() for grad in task2_g], dim=0)
                    shared_g = CAGrad(torch.stack([task1_g, task2_g]), args.c)

                    shared_grad_flatten = shared_g + shared_h

                elif args.method == "a-cagrad":
                    task1_shared_grad = torch.cat(
                        [grad.flatten() for grad in task1_shared_grad], dim=0
                    )
                    task2_shared_grad = torch.cat(
                        [grad.flatten() for grad in task2_shared_grad], dim=0
                    )
                    shared_grad_flatten = CAGrad(
                        torch.stack([task1_shared_grad, task2_shared_grad]), args.c
                    )

                shared_grad_id = 0
                shared_grad = []
                for name, param in model.named_parameters():
                    if "task" not in name:
                        length = param.grad.flatten().shape[0]
                        shared_grad.append(
                            shared_grad_flatten[
                                shared_grad_id : shared_grad_id + length
                            ].reshape(param.shape)
                        )
                        shared_grad_id += length

            elif args.method == "imtl":
                # IMTL

                task1_h = torch.cat([grad.flatten() for grad in task1_h], dim=0)
                task2_h = torch.cat([grad.flatten() for grad in task2_h], dim=0)
                shared_h = IMTL([task1_h, task2_h])

                task1_g = torch.cat([grad.flatten() for grad in task1_g], dim=0)
                task2_g = torch.cat([grad.flatten() for grad in task2_g], dim=0)
                shared_g = IMTL([task1_g, task2_g])

                shared_grad_flatten = shared_g + shared_h

                shared_grad_id = 0
                shared_grad = []
                for name, param in model.named_parameters():
                    if "task" not in name:
                        length = param.grad.flatten().shape[0]
                        shared_grad.append(
                            shared_grad_flatten[
                                shared_grad_id : shared_grad_id + length
                            ].reshape(param.shape)
                        )
                        shared_grad_id += length
        index_w = 0
        index_shared_grad = 0
        task1_index_classifier_grad = 0
        task2_index_classifier_grad = 0
        for name, param in model.named_parameters():
            param.data = old_w[index_w]
            index_w += 1

            if "task_1" in name:
                param.grad.data = task1_classifier_grad[task1_index_classifier_grad]
                task1_index_classifier_grad += 1
            elif "task_2" in name:
                param.grad.data = task2_classifier_grad[task2_index_classifier_grad]
                task2_index_classifier_grad += 1
            elif "task" not in name:
                param.grad.data = shared_grad[index_shared_grad]
                index_shared_grad += 1
            else:
                raise ValueError(f"Unknown layer {name}")

        assert index_w == len(old_w), f"Redundant gradient: {index_w} vs {len(old_w)}"
        assert index_shared_grad == len(
            shared_grad
        ), f"Redundant gradient: {index_shared_grad} vs {len(shared_grad)}"
        assert task1_index_classifier_grad == len(
            task1_classifier_grad
        ), f"Redundant gradient: {task1_index_classifier_grad} vs {len(task1_classifier_grad)}"
        assert task2_index_classifier_grad == len(
            task2_classifier_grad
        ), f"Redundant gradient: {task2_index_classifier_grad} vs {len(task2_classifier_grad)}"

        shared_optimizer.step()
        classifier_optimizer.step()
        model.zero_grad()


@torch.no_grad()
def test():

    model.eval()

    acc_1 = 0
    acc_2 = 0

    with torch.no_grad():

        for (it, batch) in enumerate(test_loader):
            X = batch[0]
            y = batch[1]
            X = X.cuda()
            y = y.cuda()

            out1_prob, out2_prob = model(X)
            out1_prob = F.softmax(out1_prob, dim=1)
            out2_prob = F.softmax(out2_prob, dim=1)
            out1 = out1_prob.max(1)[1]
            out2 = out2_prob.max(1)[1]
            acc_1 += (out1 == y[:, 0]).sum()
            acc_2 += (out2 == y[:, 1]).sum()

        acc_1 = acc_1.item() / len(test_loader.dataset)
        acc_2 = acc_2.item() / len(test_loader.dataset)

    return acc_1, acc_2


args.rho_eval = [float(rho) for rho in args.rho_eval]
best_acc1 = 0
best_acc2 = 0
best_acc = 0
surrogate_loss_1 = {rho: [] for rho in args.rho_eval}
surrogate_loss_2 = {rho: [] for rho in args.rho_eval}
grad_norm_1 = {rho: [] for rho in args.rho_eval}
grad_norm_2 = {rho: [] for rho in args.rho_eval}
for i in range(args.n_epochs):
    logging.info(f"Epoch [{i}/{args.n_epochs}]")
    losses = train(i)

    acc_1, acc_2 = test()
    logging.info(f"Accuracy task 1: {acc_1}, Accuracy task 2: {acc_2}")
    acc = (acc_1 + acc_2) / 2
    if acc > best_acc:
        log_str = f"Score improved from {best_acc} to {acc}. Saving model to {args.output_dir}"
        logging.info(log_str)
        torch.save(model.state_dict(), f"{args.output_dir}/model.pt")
        best_acc = acc
torch.save(surrogate_loss_1, f"{args.output_dir}/surrogate_loss_1.pt")
torch.save(surrogate_loss_2, f"{args.output_dir}/surrogate_loss_2.pt")
torch.save(grad_norm_1, f"{args.output_dir}/grad_norm_1.pt")
torch.save(grad_norm_2, f"{args.output_dir}/grad_norm_2.pt")
