import numpy as np
import random
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def write_log(log, log_path):
    f = open(log_path, mode="a")
    f.write(str(log))
    f.write("\n")
    f.close()
