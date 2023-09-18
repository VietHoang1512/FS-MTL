import re
import glob
import numpy as np


def str2float(list_str):
    return [float(x) for x in list_str][-10:]


patterns = [
    "Test Mean IoU:",
    "Test Pixel Accuracy:",
    "Test Absolute Error:",
    "Test Relative Error:",
    "Test Loss Mean:",
    "Test Loss Med:",
    "Test Loss <11.25:",
    "Test Loss <22.5:",
    "Test Loss <30:",
    "Test ∆m,",
]

for pattern in patterns:
    for fp in glob.glob(
        "/home/ubuntu/implement/MOO-SAM/nyuv2/method=pcgrad/model=mtan/n_epochs=200/rho=0.05/**/train.log",
        recursive=True,
    ):

        res = []
        with open(fp, "r") as f:
            res_txt = f.read()
        res.append(np.mean(str2float(re.findall(pattern + " ([\d\.-]+)\n", res_txt))))

    print(pattern, np.mean(res) * 100)
