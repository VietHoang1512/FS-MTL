import re
import glob
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from model_resnet import ResnetModel

import seaborn as sns
import matplotlib.pyplot as plt
from reliability_diagrams import reliability_diagrams


batch_size = 256
DATA_PATH = "data"


def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
    """Collects predictions into bins used to draw a reliability diagram.
    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.
    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.
    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert len(confidences) == len(pred_labels)
    assert len(confidences) == len(true_labels)
    assert num_bins > 0

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)

    return ece


def brier_score(y_true, y):
    y_pred = y.cpu().numpy()
    return (
        1
        + (np.sum(y_pred ** 2) - 2 * np.sum(y_pred[np.arange(y_pred.shape[0]), y_true]))
        / y_true.shape[0]
    )


def entropy_loss(x):
    out = x * torch.log(x)
    out = -1.0 * out.sum(dim=1)
    return out


datasets = ["multi_mnist", "multi_fashion", "multi_fashion_and_mnist"]

result = {
    "Task1": {dset: dict() for dset in datasets},
    "Task2": {dset: dict() for dset in datasets},
}
plot = {dset: dict() for dset in datasets}
for path in glob.glob(
    "/home/ubuntu/implement/MOO-SAM/multi-mnist/adaptive_eval=False/batch_size=256/c=0.4/dset=*/lr=0.001/method=cagrad/n_epochs=200/rho=*.0/rho_eval=[.01/.05]/seed=0/model.pt",
    recursive=True,
):
    # if "multi_fashion_and_mnist" in path:
    #     continue
    print("Loading model from {}".format(path))
    model = ResnetModel(2)
    state_dict = torch.load(path, map_location="cpu")

    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    src = re.findall("dset=(.+)/lr=", path)[0]
    method = "ERM" if "n_epochs=200/rho=0.0/" in path else "Ours"
    for tgt in datasets:

        dataset_fp = f"{DATA_PATH}/{tgt}.pickle"
        with open(dataset_fp, "rb") as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

        testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
        testLabel = torch.from_numpy(testLabel).long()

        test_set = torch.utils.data.TensorDataset(testX, testLabel)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False
        )

        with torch.no_grad():

            task1_ground_truth = []
            task2_ground_truth = []
            task1_preds = []
            task2_preds = []
            task1_preds = []
            task2_preds = []
            with torch.no_grad():

                for (it, batch) in enumerate(test_loader):
                    X = batch[0]
                    y = batch[1]
                    X = X.cuda()
                    y = y.cuda()

                    out1_prob, out2_prob = model(X)
                    out1_prob = F.softmax(out1_prob, dim=1)
                    out2_prob = F.softmax(out2_prob, dim=1)

                    task1_ground_truth.append(y[:, 0].detach().clone())
                    task2_ground_truth.append(y[:, 1].detach().clone())
                    task1_preds.append(out1_prob.detach().clone())
                    task2_preds.append(out2_prob.detach().clone())

                task1_preds = torch.cat(task1_preds)
                task2_preds = torch.cat(task2_preds)

                task1_ground_truth = torch.cat(task1_ground_truth).cpu().numpy()
                task2_ground_truth = torch.cat(task2_ground_truth).cpu().numpy()

                task1_brier = brier_score(task1_ground_truth, task1_preds)
                task2_brier = brier_score(task2_ground_truth, task2_preds)
                task1_entropy = entropy_loss(task1_preds)
                task2_entropy = entropy_loss(task2_preds)

                task1_entropy = task1_entropy.cpu().numpy()
                task2_entropy = task2_entropy.cpu().numpy()
                result["Task1"][tgt][method + "_" + src] = task1_entropy
                result["Task2"][tgt][method + "_" + src] = task2_entropy
                if src == tgt:
                    print("*" * 20)
                    print(src, method)
                    print("Brier")
                    print(f"{task1_brier:.3f}", f"{task2_brier:.3f}")
                    task1_confidence, task1_preds = torch.max(task1_preds, 1)
                    task2_confidence, task2_preds = torch.max(task2_preds, 1)
                    print("ECE")
                    print(
                        f"{compute_calibration(task1_ground_truth, task1_preds.cpu().numpy(), task1_confidence.cpu().numpy()):.3f}",
                        f"{compute_calibration(task2_ground_truth, task2_preds.cpu().numpy(), task2_confidence.cpu().numpy()):.3f}",
                    )

                    plot[tgt][method] = dict()
                    plot[tgt][method]["true_labels"] = np.concatenate(
                        [task1_ground_truth, task2_ground_truth]
                    )
                    plot[tgt][method]["pred_labels"] = np.concatenate(
                        [task1_preds.cpu().numpy(), task2_preds.cpu().numpy()]
                    )
                    plot[tgt][method]["confidences"] = np.concatenate(
                        [task1_confidence.cpu().numpy(), task2_confidence.cpu().numpy()]
                    )


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

fig = plt.subplot(2, 2, 1)
for key, val in result["Task1"]["multi_fashion"].items():
    if not key.endswith("multi_fashion"):
        if not key.startswith("Our"):
            sns.kdeplot(val, label=key, linestyle="--")
        else:
            sns.kdeplot(val, label=key, linestyle="-")
plt.legend(fontsize=18)
plt.grid()
fig.axes.get_xaxis().get_label().set_visible(False)
fig.axes.get_yaxis().get_label().set_visible(False)

fig = plt.subplot(2, 2, 3)
for key, val in result["Task2"]["multi_fashion"].items():
    if not key.endswith("multi_fashion"):
        if not key.startswith("Our"):
            sns.kdeplot(val, label=key, linestyle="-")
        else:
            sns.kdeplot(val, label=key, linestyle="-")
plt.legend(fontsize=18)
plt.grid()
fig.axes.get_xaxis().get_label().set_visible(False)
fig.axes.get_yaxis().get_label().set_visible(False)


fig = plt.subplot(2, 2, 2)
for key, val in result["Task1"]["multi_mnist"].items():
    if not key.endswith("multi_mnist"):
        if not key.startswith("Our"):
            sns.kdeplot(val, label=key, linestyle="--")
        else:
            sns.kdeplot(val, label=key, linestyle="-")
plt.legend(fontsize=18)
plt.grid()
fig.axes.get_xaxis().get_label().set_visible(False)
fig.axes.get_yaxis().get_label().set_visible(False)

fig = plt.subplot(2, 2, 4)
for key, val in result["Task2"]["multi_mnist"].items():
    if not key.endswith("multi_mnist"):
        if not key.startswith("Our"):
            sns.kdeplot(val, label=key, linestyle="--")
        else:
            sns.kdeplot(val, label=key, linestyle="-")
plt.legend(fontsize=18)
plt.grid()
fig.axes.get_xaxis().get_label().set_visible(False)
fig.axes.get_yaxis().get_label().set_visible(False)

plt.tight_layout()
plt.savefig(f"out-entropy.pdf")
plt.close()


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

fig = plt.subplot(2, 2, 1)
for key, val in result["Task1"]["multi_fashion"].items():
    if key.endswith("multi_fashion"):
        if not key.startswith("Our"):
            sns.kdeplot(val, label=key, linestyle="--")
        else:
            sns.kdeplot(val, label=key, linestyle="-")
plt.legend(fontsize=18)
plt.grid()
fig.axes.get_xaxis().get_label().set_visible(False)
fig.axes.get_yaxis().get_label().set_visible(False)

fig = plt.subplot(2, 2, 3)
for key, val in result["Task2"]["multi_fashion"].items():
    if key.endswith("multi_fashion"):
        if not key.startswith("Our"):
            sns.kdeplot(val, label=key, linestyle="--")
        else:
            sns.kdeplot(val, label=key, linestyle="-")
plt.legend(fontsize=18)
plt.grid()
fig.axes.get_xaxis().get_label().set_visible(False)
fig.axes.get_yaxis().get_label().set_visible(False)


fig = plt.subplot(2, 2, 2)
for key, val in result["Task1"]["multi_mnist"].items():
    if key.endswith("multi_mnist"):
        if not key.startswith("Our"):
            sns.kdeplot(val, label=key, linestyle="--")
        else:
            sns.kdeplot(val, label=key, linestyle="-")
plt.legend(fontsize=18)
plt.grid()
fig.axes.get_xaxis().get_label().set_visible(False)
fig.axes.get_yaxis().get_label().set_visible(False)

fig = plt.subplot(2, 2, 4)
for key, val in result["Task2"]["multi_mnist"].items():
    if key.endswith("multi_mnist"):
        if not key.startswith("Our"):
            sns.kdeplot(val, label=key, linestyle="-")
        else:
            sns.kdeplot(val, label=key, linestyle="-")
plt.legend(fontsize=18)
plt.grid()
fig.axes.get_xaxis().get_label().set_visible(False)
fig.axes.get_yaxis().get_label().set_visible(False)

plt.tight_layout()
plt.savefig(f"in-entropy.pdf")
plt.close()


results = dict()
for i, dset in enumerate(datasets):
    for method in ["ERM", "Ours"]:
        key = f"{method}" if i == 0 else f"{method}_{dset}"
        results[key] = plot[dset][method]
fig = reliability_diagrams(
    results,
    num_bins=10,
    draw_ece=True,
    draw_bin_importance=False,
    num_cols=2,
    dpi=72,
    return_fig=True,
)
plt.tight_layout()
plt.savefig(f"ECE.pdf")
plt.close()
