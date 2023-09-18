import argparse
import pickle
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_resnet import ResnetModel
from utils import setup_seed


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_net_parameters(net):
    params = []
    for _, param in net.named_parameters():
        params.append(param.data.clone().flatten())
    return torch.cat(params)


def get_noised_model(net, weight, noise_vector):
    new_weight = (weight + noise_vector).data.clone()
    cur_length = 0
    for _, param in net.named_parameters():
        length = param.numel()
        param.data = new_weight[cur_length : cur_length + length].reshape(param.shape)
        cur_length += length
    assert cur_length == len(new_weight)
    return net


criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(net, loader):

    net.eval()

    acc_1 = 0
    acc_2 = 0
    loss_1 = 0
    loss_2 = 0
    with torch.no_grad():

        for (it, batch) in enumerate(loader):
            X = batch[0]
            y = batch[1]
            X = X.cuda()
            y = y.cuda()

            out1_prob, out2_prob = net(X)
            loss1 = criterion(out1_prob, y[:, 0])
            loss2 = criterion(out2_prob, y[:, 1])
            out1_prob = F.softmax(out1_prob, dim=1)
            out2_prob = F.softmax(out2_prob, dim=1)
            out1 = out1_prob.max(1)[1]
            out2 = out2_prob.max(1)[1]
            acc_1 += (out1 == y[:, 0]).sum()
            acc_2 += (out2 == y[:, 1]).sum()
            loss_1 += loss1.item()
            loss_2 += loss2.item()

        acc_1 = 100 * acc_1.item() / len(loader.dataset)
        acc_2 = 100 * acc_2.item() / len(loader.dataset)
        loss_1 /= len(loader.dataset)
        loss_2 /= len(loader.dataset)
    return acc_1, acc_2, loss_1, loss_2


parser = argparse.ArgumentParser(description="plotting loss surface")
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

parser.add_argument("--model_file", default="", help="path to the trained model file")

parser.add_argument("--seed", type=int, default=0, help="seed")

args = parser.parse_args()

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

model = ResnetModel(2)
model.load_state_dict(torch.load(args.model_file, map_location="cpu"))
model.cuda()

n_params = count_parameters(model)

net_param = get_net_parameters(model)
gamma_list = torch.arange(start=0, end=1000, step=50)
# ori_acc_1, ori_acc_2, ori_loss_1, ori_loss_2 = evaluate(model, train_loader)

TRAIN_ACC_1 = []
TRAIN_ACC_1_STD = []
TRAIN_ACC_2 = []
TRAIN_ACC_2_STD = []

TEST_ACC_1 = []
TEST_ACC_1_STD = []
TEST_ACC_2 = []
TEST_ACC_2_STD = []

for gamma in tqdm(gamma_list, total=len(gamma_list)):
    train_gamma_acc_1 = []
    train_gamma_acc_2 = []

    test_gamma_acc_1 = []
    test_gamma_acc_2 = []

    for _ in range(10):
        direction_vector = torch.randn(n_params).cuda()
        unit_direction_vector = direction_vector / torch.norm(direction_vector)
        noised_model = get_noised_model(model, net_param, unit_direction_vector * gamma)
        train_acc_1, train_acc_2, train_loss_1, train_loss_2 = evaluate(
            noised_model, train_loader
        )
        test_acc_1, test_acc_2, test_loss_1, test_loss_2 = evaluate(
            noised_model, test_loader
        )

        train_gamma_acc_1.append(train_acc_1)
        train_gamma_acc_2.append(train_acc_2)

        test_gamma_acc_1.append(test_acc_1)
        test_gamma_acc_2.append(test_acc_2)
    TRAIN_ACC_1.append(np.mean(train_gamma_acc_1))
    TRAIN_ACC_1_STD.append(np.std(train_gamma_acc_1))
    TRAIN_ACC_2.append(np.mean(train_gamma_acc_2))
    TRAIN_ACC_2_STD.append(np.std(train_gamma_acc_2))

    TEST_ACC_1.append(np.mean(test_gamma_acc_1))
    TEST_ACC_1_STD.append(np.std(test_gamma_acc_1))
    TEST_ACC_2.append(np.mean(test_gamma_acc_2))
    TEST_ACC_2_STD.append(np.std(test_gamma_acc_2))

with open(args.model_file + "_log.txt", "w") as f:
    f.write(f"TRAIN_TASK_1={TRAIN_ACC_1}\n")
    f.write(f"TRAIN_TASK_1_STD={TRAIN_ACC_1_STD}\n")
    f.write(f"TRAIN_TASK_2={TRAIN_ACC_2}\n")
    f.write(f"TRAIN_TASK_2_STD={TRAIN_ACC_2_STD}\n")

    f.write(f"TEST_TASK_1={TEST_ACC_1}\n")
    f.write(f"TEST_TASK_1_STD={TEST_ACC_1_STD}\n")
    f.write(f"TEST_TASK_2={TEST_ACC_2}\n")
    f.write(f"TEST_TASK_2_STD={TEST_ACC_2_STD}\n")
