import torch
import scipy.stats as stats
import numpy as np

from numpy import median, shape, sqrt
from numpy.random import permutation
from scipy.spatial.distance import pdist, squareform

def data_normalize(data):
    data = stats.zscore(data, ddof=1, axis=0)
    data[np.isnan(data)] = 0.
    return data

def reduce_func(K, thresh, need_wx=False):
    n = K.shape[0]
    wx, vx = np.linalg.eigh(0.5 * (K + K.T))
    topkx = int(np.min((400, np.floor(n / 4))))
    idx = np.argsort(-wx)
    wx = wx[idx]
    vx = vx[:, idx]
    wx = wx[0:topkx]
    vx = vx[:, 0:topkx]
    vx = vx[:, wx > wx.max() * thresh]
    wx = wx[wx > wx.max() * thresh]
    # vx = 2 * np.sqrt(n) * vx.dot(np.diag(np.sqrt(wx))) / np.sqrt(wx[0])
    if need_wx:
        return 2 * np.sqrt(n) * vx.dot(np.diag(np.sqrt(wx))) / np.sqrt(wx[0])
    else:
        return vx.dot(np.diag(np.sqrt(wx)))

def cal_kernel(X, length, Y=None):
    Xsq = (X ** 2).sum(dim=1, keepdim=True)
    if Y is None:
        sqdist = Xsq + Xsq.T - 2*X.mm(X.T)
    else:
        Ysq = (Y ** 2).sum(dim=1, keepdim=True)
        sqdist = Xsq + Ysq.T - 2 * X.mm(Y.T)
    return torch.exp(- 0.5 * sqdist / (length**2))

def Pdist2(x):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y = x
    y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def data_split(z, x, y, training_size = 0.5):
    n = z.shape[0]
    if training_size == 0:
        z_tr = z_te = z
        x_tr = x_te = x
        y_tr = y_te = y
    else:
        training_set_per = training_size
        idx_tr = np.random.choice(n, int(training_set_per*n), replace=False)
        idx_te = np.delete(np.arange(n), idx_tr)
        z_tr = z[idx_tr, :]; x_tr = x[idx_tr, :]; y_tr = y[idx_tr, :]
        z_te = z[idx_te, :]; x_te = x[idx_te, :]; y_te = y[idx_te, :]

    z_tr, x_tr, y_tr = totensor(z_tr, x_tr, y_tr)
    z_te, x_te, y_te = totensor(z_te, x_te, y_te)
    return z_tr, z_te, x_tr, x_te, y_tr, y_te

def totensor(z, x, y, device="cpu"):
    if isinstance(x, np.ndarray):
        z = torch.from_numpy(z)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
    z = z.to(device)
    x = x.to(device)
    y = y.to(device)
    return z, x, y

def Continuous2Discrete(Data_dir=None, bins_nums=20):
    Data_dir['threshold'] = 0.01
    Data_dir['width_init'] = 0.01
    data = Data_dir['data_mat']
    for i in range(data.shape[-1]):
        data_c = data[:, i]
        max = np.max(data_c)
        min = np.min(data_c)
        bin_nums = bins_nums
        bins = np.linspace(min, max, num=bin_nums)
        data_d = np.digitize(data_c, bins=bins)
        data[:, i] = data_d
    return data