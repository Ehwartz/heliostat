import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    a = torch.ones(size=[4, 2])
    b = torch.ones(size=[4])
    a[:, 0] = a[:, 0] + b
    print(a)