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
from utils import setup_seed
from bypass_bn import enable_running_stats, disable_running_stats


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

parser.add_argument(
    "--adaptive",
    default=False,
    type=str2bool,
    help="True if you want to use the Adaptive SAM.",
)

parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")

parser.add_argument("--seed", type=int, default=0, help="seed")

args = parser.parse_args()

args.output_dir = "outputs-single-task-1/" + str(args).replace(", ", "/").replace(
    "'", ""
).replace("(", "").replace(")", "").replace("Namespace", "")

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

    for (it, batch) in tqdm(
        enumerate(train_loader),
        desc=f"Training on epoch [{epoch}/{args.n_epochs}]",
        total=len(train_loader),
    ):

        X = batch[0]
        y = batch[1]
        X, y = X.cuda(), y.cuda()
        batchsize_cur = X.shape[0]

        model.train()
        model.zero_grad()

        enable_running_stats(model)
        out1, _ = model(X)

        ##### SAM stage 1, task 1 #####
        loss1 = criterion(out1, y[:, 0])

        loss1.backward(retain_graph=True)

        task1_norms = []
        task1_ew = []
        task1_old_w = []
        old_w = []
        for name, param in model.named_parameters():
            old_w.append(param.data.clone())
            if "task_2" not in name:
                task1_norms.append(
                    (
                        ((torch.abs(param) if args.adaptive else 1.0) * param.grad)
                        .norm(p=2)
                        .data.clone()
                    )
                )
                ew = (torch.pow(param, 2) if args.adaptive else 1.0) * param.grad
                task1_ew.append(ew.data.clone().flatten())
                task1_old_w.append(param.data.clone().flatten())
                param.grad.zero_()

        task1_norm = torch.norm(torch.stack(task1_norms), p=2)
        task1_scale = args.rho / (task1_norm + 1e-12)

        task1_ew = torch.cat(task1_ew, dim=0) * task1_scale
        task1_old_w = torch.cat(task1_old_w, dim=0)

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
                    param.grad.zero_()
            else:
                task1_shared_grad.append(param.grad.detach().data.clone().flatten())
                param.grad.zero_()

        task1_shared_grad = torch.cat(task1_shared_grad, dim=0)

        shared_grad = task1_shared_grad

        index_w = 0
        index_shared_grad = 0
        task1_index_classifier_grad = 0
        for name, param in model.named_parameters():
            param.data = old_w[index_w]
            index_w += 1

            if "task_1" in name:
                param.grad.data = task1_classifier_grad[task1_index_classifier_grad]
                task1_index_classifier_grad += 1
            elif "task_2" in name:
                continue
            elif "task" not in name:
                length = param.grad.flatten().shape[0]
                param.grad.data = shared_grad[
                    index_shared_grad : index_shared_grad + length
                ].reshape(param.grad.shape)
                index_shared_grad += length
            else:
                raise ValueError(f"Unknown layer {name}")

        assert index_w == len(old_w), f"Redundant gradient: {index_w} vs {len(old_w)}"
        assert index_shared_grad == len(
            shared_grad
        ), f"Redundant gradient: {index_shared_grad} vs {len(shared_grad)}"
        assert task1_index_classifier_grad == len(
            task1_classifier_grad
        ), f"Redundant gradient: {task1_index_classifier_grad} vs {len(task1_classifier_grad)}"

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


best_acc1 = 0
best_acc2 = 0
best_acc = 0
for i in range(args.n_epochs):
    logging.info(f"Epoch [{i}/{args.n_epochs}]")
    losses = train(i)

    acc_1, acc_2 = test()
    logging.info(f"Accuracy task 1: {acc_1}, Accuracy task 2: {acc_2}")
    acc = acc_1
    if acc > best_acc:
        log_str = f"Score improved from {best_acc} to {acc}. Saving model to {args.output_dir}"
        logging.info(log_str)
        # torch.save(model.state_dict(), f"{args.output_dir}/model.pt")
        best_acc = acc
