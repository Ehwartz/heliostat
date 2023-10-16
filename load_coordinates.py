import numpy as np
import torch
import pandas as pd

if __name__ == '__main__':
    df = pd.read_excel('./coordinates.xlsx', header=0)
    xys = torch.from_numpy(np.array(df))

    # xys = torch.load('./xys.pth')
    torch.save(xys, './xys.pth')
    print(xys)

    print(xys.shape)

