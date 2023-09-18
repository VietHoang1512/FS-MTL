import os
import sys
import yaml
import logging
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

from data import NYUv2
from models import SegNet, SegNetMtan
from utils import ConfMatrix, delta_fn, depth_error, normal_error
from utils import (
    common_parser,
    set_seed,
    str2bool,
)
from bypass_bn import disable_running_stats, enable_running_stats

from mtl import PCGrad, CAGrad, MinNormSolver, IMTL


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    if task_type == "normal":
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss


# args = 0
def main(path, lr, bs, device):
    # ----
    # Nets
    # ---
    model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    model = model.to(device)

    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on NYUv2."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    nyuv2_train_set = NYUv2(
        root=path.as_posix(), train=True, augmentation=args.apply_augmentation
    )
    nyuv2_test_set = NYUv2(root=path.as_posix(), train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set, batch_size=bs, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set, batch_size=bs, shuffle=False
    )

    # optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=lr),
        ],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    epochs = args.n_epochs

    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 24], dtype=np.float32)
    custom_step = -1
    conf_mat = ConfMatrix(model.segnet.class_nb)
    TASKS = [0, 1, 2]
    N_TASKS = len(TASKS)

    for epoch in range(epochs):
        logging.info(f"[Epoch {epoch}/{epochs}]")
        cost = np.zeros(24, dtype=np.float32)
        epoch_iter = tqdm(enumerate(train_loader), total=train_batch)
        for j, batch in epoch_iter:
            custom_step += 1

            model.train()
            optimizer.zero_grad()

            train_data, train_label, train_depth, train_normal = batch
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            enable_running_stats(model)
            train_pred, features = model(train_data, return_representation=True)

            losses = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic"),
                    calc_loss(train_pred[1], train_depth, "depth"),
                    calc_loss(train_pred[2], train_normal, "normal"),
                )
            )

            ##### SAM, stage 1 #####
            epsilon_weights = dict()
            old_weights = dict()
            epsilon_weights = {"rep": dict()}
            old_weights["rep"] = []

            shared_g = dict()

            for task in TASKS:
                model.zero_grad()
                losses[task].backward(retain_graph=True)
                shared_norms = []
                epsilon_weights["rep"][task] = []
                task_norms = []
                old_weights[task] = []
                epsilon_weights[task] = []
                shared_g[task] = []

                for name, param in model.segnet.named_parameters():

                    if name.startswith("pred_task"):
                        if name.startswith("pred_task{}".format(task + 1)):
                            ew = torch.zeros_like(param).data.clone()
                            if param.grad is not None:
                                task_norms.append(
                                    (
                                        (torch.abs(param) if args.adaptive else 1.0)
                                        * param.grad
                                    )
                                    .norm(p=2)
                                    .data.clone()
                                )
                                old_weights[task].append(param.data.clone())

                                ew = (
                                    (torch.pow(param, 2) if args.adaptive else 1.0)
                                    * param.grad
                                ).data.clone()
                            epsilon_weights[task].append(ew)
                    else:
                        ew = torch.zeros_like(param).data.clone()
                        if param.grad is not None:
                            shared_norms.append(
                                (
                                    (torch.abs(param) if args.adaptive else 1.0)
                                    * param.grad
                                )
                                .norm(p=2)
                                .data.clone()
                            )
                            shared_g[task].append(param.grad.data.clone())
                            ew = (
                                (torch.pow(param, 2) if args.adaptive else 1.0)
                                * param.grad
                            ).data.clone()
                        epsilon_weights["rep"][task].append(ew)
                        if task == 0:
                            old_weights["rep"].append(param.data.clone())

                task_norm = torch.norm(torch.stack(task_norms), p=2)
                task_scale = (args.rho / (task_norm + 1e-12)).item()
                epsilon_weights[task] = [
                    epsilon_weight * task_scale
                    for epsilon_weight in epsilon_weights[task]
                ]

                shared_norm = torch.norm(torch.stack(shared_norms), p=2)
                shared_scale = (args.rho / (shared_norm + 1e-12)).item()
                epsilon_weights["rep"][task] = [
                    epsilon_weight * shared_scale
                    for epsilon_weight in epsilon_weights["rep"][task]
                ]
            del task_norms, shared_norms
            # logging.info("DONE STAGE 1")
            ##### SAM, stage 2 #####
            disable_running_stats(model)
            SAM_gradient = dict()
            total = len(old_weights["rep"])
            model.zero_grad()
            for task in TASKS:
                assert (
                    len(epsilon_weights["rep"][task]) == total
                ), f"Encoder weight's length not equal: {len(epsilon_weights['rep'][task])} vs {total}"

                assert len(epsilon_weights[task]) == len(
                    old_weights[task]
                ), f"Decoder weight's length not equal: {len(epsilon_weights[task])} vs {len(old_weights[task])}"

                shared_idx = 0
                classifier_idx = 0

                for name, param in model.segnet.named_parameters():
                    if name.startswith("pred_task"):
                        if name.startswith("pred_task{}".format(task + 1)):
                            param.data = (
                                old_weights[task][classifier_idx]
                                + epsilon_weights[task][classifier_idx]
                            ).data.clone()
                            classifier_idx += 1
                    else:
                        if param.grad is not None:
                            param.grad.zero_()
                        param.data = (
                            old_weights["rep"][shared_idx]
                            + epsilon_weights["rep"][task][shared_idx]
                        ).data.clone()
                        shared_idx += 1
                assert (
                    shared_idx == total
                ), f"Encoder weight's length not equal: {shared_idx} vs {total}"
                assert classifier_idx == len(
                    old_weights[task]
                ), f"Classifier weight's length not equal: {classifier_idx} vs {len(old_weights[task])}"

                train_pred, _ = model(train_data, return_representation=True)

                if task == 0:
                    calc_loss(train_pred[0], train_label, "semantic").backward()
                elif task == 1:
                    calc_loss(train_pred[1], train_depth, "depth").backward()
                elif task == 2:
                    calc_loss(train_pred[2], train_normal, "normal").backward()
                else:
                    raise ValueError(f"Task {task} not supported")

                SAM_gradient[task] = []
                classifier_idx = 0
                shared_idx = 0
                for name, param in model.segnet.named_parameters():
                    if name.startswith("pred_task"):
                        if name.startswith(f"pred_task{task+1}"):
                            param.data = old_weights[task][classifier_idx].data.clone()
                            classifier_idx += 1
                        continue

                    if param.grad is not None:
                        SAM_gradient[task].append(param.grad.data.clone())
                        param.grad.zero_()
                    else:
                        SAM_gradient[task].append(torch.zeros_like(param).data.clone())
                    shared_idx += 1
                assert (classifier_idx) == len(
                    old_weights[task]
                ), f"Classifier weight's length not equal: {classifier_idx} vs {len(old_weights[task])}"
                assert (
                    shared_idx == total
                ), f"Encoder weight's length not equal: {shared_idx} vs {total}"

            # logging.info("DONE STAGE 2")
            del epsilon_weights
            if "mgda" in args.method:
                # MGDA:
                if args.method == "a-mgda":
                    with torch.no_grad():
                        sol, _ = MinNormSolver.find_min_norm_element(
                            [SAM_gradient[task] for task in TASKS]
                        )
                    shared_idx = 0
                    for name, param in model.segnet.named_parameters():
                        if name.startswith("pred_task"):
                            continue
                        param.grad = (
                            sum(
                                [
                                    sol[task] * SAM_gradient[task][shared_idx]
                                    for task in TASKS
                                ]
                            )
                            * N_TASKS
                        )

                        param.data = old_weights["rep"][shared_idx].data.clone()
                        shared_idx += 1

                elif args.method == "mgda":
                    shared_h = dict()
                    shared_h_norm = dict()
                    shared_g_norm = dict()
                    for task in TASKS:
                        shared_h[task] = []
                        shared_h_norm[task] = []
                        shared_g_norm[task] = []
                        shared_idx = 0
                        for name, param in model.segnet.named_parameters():
                            if name.startswith("pred_task"):
                                continue
                            shared_h[task].append(
                                SAM_gradient[task][shared_idx]
                                - shared_g[task][shared_idx]
                            )
                            shared_h_norm[task].append(
                                shared_h[task][shared_idx].norm(p=2).data.clone()
                            )
                            shared_g_norm[task].append(
                                shared_g[task][shared_idx].norm(p=2).data.clone()
                            )
                            shared_idx += 1
                        shared_h_norm[task] = torch.norm(
                            torch.stack(shared_h_norm[task]), p=2
                        )
                        shared_g_norm[task] = torch.norm(
                            torch.stack(shared_g_norm[task]), p=2
                        )
                        shared_h[task] = [
                            grad / shared_h_norm[task] for grad in shared_h[task]
                        ]
                        shared_g[task] = [
                            grad / shared_g_norm[task] for grad in shared_g[task]
                        ]

                    with torch.no_grad():
                        sol_h, _ = MinNormSolver.find_min_norm_element(
                            [shared_h[task] for task in TASKS]
                        )
                        sol_g, _ = MinNormSolver.find_min_norm_element(
                            [shared_g[task] for task in TASKS]
                        )
                    shared_idx = 0
                    for name, param in model.segnet.named_parameters():
                        if name.startswith("pred_task"):
                            continue
                        param.grad = (
                            sum(
                                [
                                    (
                                        sol_h[task] * shared_h[task][shared_idx]
                                        + sol_g[task] * shared_g[task][shared_idx]
                                    )
                                    for task in TASKS
                                ]
                            )
                            * N_TASKS
                        )
                        param.data = old_weights["rep"][shared_idx].data.clone()
                        shared_idx += 1
            elif "pcgrad" in args.method:
                # PCGrad:
                if args.method == "pcgrad":
                    shared_h = dict()
                    for task in TASKS:
                        shared_h[task] = []
                        shared_idx = 0
                        for name, param in model.segnet.named_parameters():
                            if name.startswith("pred_task"):
                                continue
                            shared_h[task].append(
                                SAM_gradient[task][shared_idx]
                                - shared_g[task][shared_idx]
                            )
                            shared_idx += 1
                    with torch.no_grad():
                        shared_g = PCGrad([shared_g[task] for task in TASKS])
                        shared_h = PCGrad([shared_h[task] for task in TASKS])
                    shared_idx = 0
                    for name, param in model.segnet.named_parameters():
                        if name.startswith("pred_task"):
                            continue
                        param.grad = shared_g[shared_idx] + shared_h[shared_idx]
                        param.data = old_weights["rep"][shared_idx].data.clone()
                        shared_idx += 1
                elif args.method == "a-pcgrad":
                    with torch.no_grad():
                        SAM_gradient = PCGrad([SAM_gradient[task] for task in TASKS])
                    shared_idx = 0
                    for name, param in model.segnet.named_parameters():
                        if name.startswith("pred_task"):
                            continue
                        param.grad = SAM_gradient[shared_idx]
                        param.data = old_weights["rep"][shared_idx].data.clone()
                        shared_idx += 1

            elif "cagrad" in args.method:
                # CAGrad:
                if args.method == "cagrad":
                    shared_h = dict()
                    for task in TASKS:
                        shared_h[task] = []
                        shared_idx = 0
                        for name, param in model.segnet.named_parameters():
                            if name.startswith("pred_task"):
                                continue
                            shared_g[task][shared_idx] = shared_g[task][
                                shared_idx
                            ].flatten()
                            SAM_gradient[task][shared_idx] = SAM_gradient[task][
                                shared_idx
                            ].flatten()
                            shared_h[task].append(
                                SAM_gradient[task][shared_idx]
                                - shared_g[task][shared_idx]
                            )
                            shared_idx += 1
                        shared_g[task] = torch.cat(shared_g[task])
                        shared_h[task] = torch.cat(shared_h[task])
                    with torch.no_grad():
                        shared_g = CAGrad(
                            torch.stack([shared_g[task] for task in TASKS]), args.c
                        )
                        shared_h = CAGrad(
                            torch.stack([shared_h[task] for task in TASKS]), args.c
                        )
                    shared_idx = 0
                    total_length = 0
                    for name, param in model.segnet.named_parameters():
                        if name.startswith("pred_task"):
                            continue
                        length = param.data.numel()
                        param.grad = (
                            shared_g[total_length : total_length + length]
                            + shared_h[total_length : total_length + length]
                        ).reshape(param.data.size())
                        param.data = old_weights["rep"][shared_idx].data.clone()
                        total_length += length
                        shared_idx += 1

                elif args.method == "a-cagrad":

                    for task in TASKS:
                        SAM_gradient[task] = torch.cat(
                            [grad.flatten() for grad in SAM_gradient[task]]
                        )

                    with torch.no_grad():
                        SAM_gradient = CAGrad(
                            torch.stack([SAM_gradient[task] for task in TASKS]), args.c
                        )
                    shared_idx = 0
                    total_length = 0
                    for name, param in model.segnet.named_parameters():
                        if name.startswith("pred_task"):
                            continue
                        length = param.data.numel()
                        param.grad = (
                            SAM_gradient[total_length : total_length + length]
                        ).reshape(param.data.size())
                        param.data = old_weights["rep"][shared_idx].data.clone()
                        total_length += length
                        shared_idx += 1

            elif "imtl" in args.method:
                if args.method == "imtl":
                    # IMTL:
                    shared_h = dict()
                    for task in TASKS:
                        shared_h[task] = []
                        shared_idx = 0
                        for name, param in model.segnet.named_parameters():
                            if name.startswith("pred_task"):
                                continue
                            shared_g[task][shared_idx] = shared_g[task][
                                shared_idx
                            ].flatten()
                            SAM_gradient[task][shared_idx] = SAM_gradient[task][
                                shared_idx
                            ].flatten()
                            shared_h[task].append(
                                SAM_gradient[task][shared_idx]
                                - shared_g[task][shared_idx]
                            )
                            shared_idx += 1
                        shared_g[task] = torch.cat(shared_g[task])
                        shared_h[task] = torch.cat(shared_h[task])
                    with torch.no_grad():
                        shared_g = IMTL([shared_g[task] for task in TASKS])
                        shared_h = IMTL([shared_h[task] for task in TASKS])
                    shared_idx = 0
                    total_length = 0
                    for name, param in model.segnet.named_parameters():
                        if name.startswith("pred_task"):
                            continue
                        length = param.data.numel()
                        param.grad = (
                            shared_g[total_length : total_length + length]
                            + shared_h[total_length : total_length + length]
                        ).reshape(param.data.size())
                        param.data = old_weights["rep"][shared_idx].data.clone()
                        total_length += length
                        shared_idx += 1
                elif args.method == "a-imtl":
                    for task in TASKS:
                        SAM_gradient[task] = torch.cat(
                            [grad.flatten() for grad in SAM_gradient[task]]
                        )

                    with torch.no_grad():
                        SAM_gradient = IMTL(
                            torch.stack([SAM_gradient[task] for task in TASKS])
                        )
                    shared_idx = 0
                    total_length = 0
                    for name, param in model.segnet.named_parameters():
                        if name.startswith("pred_task"):
                            continue
                        length = param.data.numel()
                        param.grad = (
                            SAM_gradient[total_length : total_length + length]
                        ).reshape(param.data.size())
                        param.data = old_weights["rep"][shared_idx].data.clone()
                        total_length += length
                        shared_idx += 1
            optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = losses[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(
                train_pred[2], train_normal
            )
            avg_cost[epoch, :12] += cost[:12] / train_batch

            epoch_iter.set_description(
                f"semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}, "
                f"normal loss: {losses[2].item():.3f}"
            )

        # scheduler
        scheduler.step()
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # todo: move evaluate to function?
        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = next(test_dataset)
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                        calc_loss(test_pred[2], test_normal, "normal"),
                    )
                )

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(
                    test_pred[2], test_normal
                )
                avg_cost[epoch, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[epoch, 13:15] = conf_mat.get_metrics()

            # Test Delta_m
            test_delta_m = delta_fn(
                avg_cost[epoch, [13, 14, 16, 17, 19, 20, 21, 22, 23]]
            )

            # logging.info results
            logging.info(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
                f"| NORMAL_LOSS MEAN MED <11.25 <22.5 <30 | ∆m (test)"
            )
            logging.info(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | {avg_cost[epoch, 6]:.4f} "
                f"{avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} {avg_cost[epoch, 9]:.4f} {avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f} || "
                f"TEST: {avg_cost[epoch, 12]:.4f} {avg_cost[epoch, 13]:.4f} {avg_cost[epoch, 14]:.4f} | "
                f"{avg_cost[epoch, 15]:.4f} {avg_cost[epoch, 16]:.4f} {avg_cost[epoch, 17]:.4f} | {avg_cost[epoch, 18]:.4f} "
                f"{avg_cost[epoch, 19]:.4f} {avg_cost[epoch, 20]:.4f} {avg_cost[epoch, 21]:.4f} {avg_cost[epoch, 22]:.4f} {avg_cost[epoch, 23]:.4f} "
                f"| {test_delta_m:.3f}"
            )

            logging.info(f"Train Semantic Loss: {avg_cost[epoch, 0]}")
            logging.info(f"Train Mean IoU: {avg_cost[epoch, 1]}")
            logging.info(f"Train Pixel Accuracy: {avg_cost[epoch, 2]}")
            logging.info(f"Train Depth Loss: {avg_cost[epoch, 3]}")
            logging.info(f"Train Absolute Error: {avg_cost[epoch, 4]}")
            logging.info(f"Train Relative Error: {avg_cost[epoch, 5]}")
            logging.info(f"Train Normal Loss: {avg_cost[epoch, 6]}")
            logging.info(f"Train Loss Mean: {avg_cost[epoch, 7]}")
            logging.info(f"Train Loss Med: {avg_cost[epoch, 8]}")
            logging.info(f"Train Loss <11.25: {avg_cost[epoch, 9]}")
            logging.info(f"Train Loss <22.5: {avg_cost[epoch, 10]}")
            logging.info(f"Train Loss <30: {avg_cost[epoch, 11]}")

            logging.info(f"Test Semantic Loss: {avg_cost[epoch, 12]}")
            logging.info(f"Test Mean IoU: {avg_cost[epoch, 13]}")
            logging.info(f"Test Pixel Accuracy: {avg_cost[epoch, 14]}")
            logging.info(f"Test Depth Loss: {avg_cost[epoch, 15]}")
            logging.info(f"Test Absolute Error: {avg_cost[epoch, 16]}")
            logging.info(f"Test Relative Error: {avg_cost[epoch, 17]}")
            logging.info(f"Test Normal Loss: {avg_cost[epoch, 18]}")
            logging.info(f"Test Loss Mean: {avg_cost[epoch, 19]}")
            logging.info(f"Test Loss Med: {avg_cost[epoch, 20]}")
            logging.info(f"Test Loss <11.25: {avg_cost[epoch, 21]}")
            logging.info(f"Test Loss <22.5: {avg_cost[epoch, 22]}")
            logging.info(f"Test Loss <30: {avg_cost[epoch, 23]}")
            logging.info(f"Test ∆m, {test_delta_m}")


if __name__ == "__main__":
    parser = ArgumentParser("NYUv2", parents=[common_parser])
    parser.set_defaults(
        data_path="dataset",
        lr=1e-4,
        n_epochs=200,
        batch_size=2,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mtan",
        choices=["segnet", "mtan"],
        help="model type",
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
            "a-imtl",
        ],
        help="MTL weight method",
    )

    parser.add_argument(
        "--adaptive",
        default=True,
        type=str2bool,
        help="True if you want to use the Adaptive SAM.",
    )

    parser.add_argument("--rho", default=2, type=float, help="Rho parameter for SAM.")

    parser.add_argument(
        "--apply-augmentation", type=str2bool, default=True, help="data augmentations"
    )

    parser.add_argument("--c", type=float, default=0.4, help="c for CAGrad alg.")
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    args.output_dir = "outputs/moo-sam/" + str(args).replace(", ", "/").replace(
        "'", ""
    ).replace("(", "").replace(")", "").replace("Namespace", "")
    os.system("rm -rf " + args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{args.output_dir}/train.log",
        level=logging.DEBUG,
        filemode="w",
        datefmt="%H:%M:%S",
        format="%(asctime)s :: %(levelname)-8s \n%(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    with open(os.path.join(args.output_dir, "config.yaml"), "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)
    logging.info("Output directory:" + args.output_dir)

    device = "cuda"
    main(path=args.data_path, lr=args.lr, bs=args.batch_size, device=device)
