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
from sam import SAM
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

args.output_dir = "outputs-single-task-1-lib/" + str(args).replace(", ", "/").replace(
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

shared_optimizer = SAM(
    model.get_shared_parameters(),
    # torch.optim.SGD,
    torch.optim.Adam,
    rho=args.rho,
    adaptive=args.adaptive,
    lr=args.lr,
    # momentum=0.9
)

classifier_optimizer = SAM(
    model.get_classifier_parameters(),
    # torch.optim.SGD,
    torch.optim.Adam,
    rho=args.rho,
    adaptive=args.adaptive,
    lr=args.lr,
    # momentum=0.9
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

        model.train()
        model.zero_grad()

        ##### SAM stage 1, task 1 #####
        enable_running_stats(model)
        out1, _ = model(X)
        loss1 = criterion(out1, y[:, 0])

        loss1.backward()
        classifier_optimizer.first_step(zero_grad=True)
        shared_optimizer.first_step(zero_grad=True)

        ##### SAM stage 2, task 1 #####
        disable_running_stats(model)
        out1, _ = model(X)
        loss1 = criterion(out1, y[:, 0])

        loss1.backward()
        classifier_optimizer.second_step(zero_grad=True)
        shared_optimizer.second_step(zero_grad=True)


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
