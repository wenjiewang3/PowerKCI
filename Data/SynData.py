import numpy as np
from utils import data_normalize

def SyntheticData(CI, dim_z, nums, seed, dim_x = 1, dim_y = 1, noise_scale=0.5, T_scale=0.2):
    np.random.seed(seed)
    [seed1, seed2] = np.random.randint(0, 100000, 2)
    Z = np.random.randn(nums, dim_z)
    if CI:
        X = nonlinear_process(Z, dim_x, nums, seed1, noise_scale)
        Y = nonlinear_process(Z, dim_y, nums, seed2, noise_scale)
    else:
        T = np.random.randn(nums, 1)
        X = nonlinear_process(Z, dim_x, nums, seed1, noise_scale)
        Y = nonlinear_process(Z, dim_y, nums, seed2, noise_scale) # , T, T_scale=T_scale
        # T = np.random.randn(nums, 1)
        # TX = T_scale*T_func(T, seed1)
        # TY = T_scale*T_func(T, seed2)
        TX = T_scale*T
        TY = T_scale*T
        X += TX
        Y += TY


    X = data_normalize(X)
    Y = data_normalize(Y)
    Z = data_normalize(Z)
    return X, Y, Z


def nonlinear_process(Z, dim_X, nums, seed, noise_scale):
    dim_Z = Z.shape[-1]
    np.random.seed(seed)

    W = np.random.randn(dim_Z, dim_X)
    X =  Z @ W

    func_id_f_X = np.random.randint(1, 8, 2)
    if func_id_f_X[0] == 1:
        X = np.sin(X*np.pi)
    elif func_id_f_X[0] == 2:
        X = np.cos(X*np.pi)
    elif func_id_f_X[0] == 3:
        X = X**2/ np.sqrt(dim_Z)
    elif func_id_f_X[0] == 4:
        X = X / np.sqrt(dim_Z)
    elif func_id_f_X[0] == 5:
        X = np.exp(X) / np.sqrt(dim_Z)
    elif func_id_f_X[0] == 6:
        X = 2**X / np.sqrt(dim_Z)

    if seed % 2 == 0:
        noise_f_X = noise_scale*np.random.randn(nums, dim_X)
    else:
        noise_f_X = noise_scale*np.random.uniform(-1,1, (nums, dim_X))

    X = X + noise_f_X
    return X


def T_func(T, seed):
    np.random.seed(seed)
    func_id_f_X = np.random.randint(1, 5, 2)
    if func_id_f_X[0] == 1:
        T = np.sin(T)
    elif func_id_f_X[0] == 2:
        T = np.cos(T)
    return T

if __name__ == '__main__':
    X, Y, Z = SyntheticData([1, 1, 7], nums=600, seed=12341, CI=False)
    # print(X.shape, Y.shape, Z.shape)



