import argparse
import numpy as np
from PowerKCIModel import power_KCI
from OrgKCIModel import org_KCI
from Data.SynData import SyntheticData
from Data.PNLData import PNLData

from utils import *
import warnings
warnings.filterwarnings('ignore')
import time

def PowerTest(Z, X, Y, opt):
    KCI = power_KCI(Z, X, Y, opt)  # decomposed version
    # KCI = org_KCI(Z, X, Y, opt)  # orignal version, with the regression (X, Z) -> X
    sel_xyz_pvalue, sel_yz_pvalue, sel_z_pvalue, med_pvalue = KCI.select_kernels()

    alpha5_selxyz = (sel_xyz_pvalue <= opt.alpha)
    alpha5_selyz = (sel_yz_pvalue <= opt.alpha)
    alpha5_selz = (sel_z_pvalue <= opt.alpha)
    alpha5_med = (med_pvalue <= opt.alpha)

    return (med_pvalue, sel_z_pvalue, sel_yz_pvalue, sel_xyz_pvalue,
            alpha5_med, alpha5_selz, alpha5_selyz, alpha5_selxyz)

def MainTest(CI, z_dim, data_nums, opt):
    seed = np.random.randint(0, 10000)
    np.random.seed(seed)

    print("seed: ", seed, ", repeat_time:", opt.repeat_times, ", test_method: ", opt.test_method, ", alpha: ", opt.alpha)
    print("data: sample data, CI: ", CI, ", z_dims: ", opt.dim_z, ", data_nums: ", data_nums, "T_scale: ", opt.T_scale, "noise_scale: ", opt.noise_scale)


    med_alpha_all = 0
    selz_alpha_all = 0
    sely_alpha_all = 0
    power_alpha_all = 0

    time_list = []
    for i in range(opt.repeat_times):

        epoch_seed = np.random.randint(0, 100000)
        X, Y, Z = SyntheticData(CI, opt.dim_z, data_nums, epoch_seed, noise_scale = opt.noise_scale, T_scale=opt.T_scale)
        # X, Y, Z = PNLData(CI, opt.dim_z, data_nums, epoch_seed, noise_scale = opt.noise_scale, T_scale=opt.T_scale)

        start_time = time.time()

        method = ["median", "sel_z", "sel_yz", "sel_all"]
        (med_pvalue, selz_pvalue, sely_pvalue, power_pvalue,
        alpha_med, alpha_selz, alpha_sely, alpha_power) = PowerTest(Z, X, Y, opt)

        end_time = time.time()
        computation_time = end_time - start_time
        time_list.append(computation_time)

        med_alpha_all += alpha_med
        selz_alpha_all += alpha_selz
        sely_alpha_all += alpha_sely
        power_alpha_all += alpha_power

        if i%opt.print_freq == 0:
            if CI == True:
                print("Type I error, idx ", i,
                      "-> ", method[0], ": (", med_alpha_all, " ", med_pvalue,
                      ") ", method[1], ": (", selz_alpha_all, " ", selz_pvalue,
                      ") ", method[2], ": (", sely_alpha_all, " ", sely_pvalue,
                      ") ", method[3], ": (", power_alpha_all, " ", power_pvalue,
                      "), time: (", round(np.mean(time_list), 3), "+", round(np.var(time_list), 4), ")")
            else:
                print("Type II error, idx ", i,
                      "-> ", method[0], ": (", i - med_alpha_all + 1, " ", med_pvalue,
                      ") ", method[1], ": (", i - selz_alpha_all + 1, " ", selz_pvalue,
                      ") ", method[2], ": (", i - sely_alpha_all + 1, " ", sely_pvalue,
                      ") ", method[3], ": (", i - power_alpha_all + 1, " ", power_pvalue,
                      "), time: (", round(np.mean(time_list), 3), "+", round(np.var(time_list), 4), ")")

    print("------------------------------------------------------------")
    print("seed: ", seed, ", repeat_time:", opt.repeat_times, ", test_method: ", opt.test_method, ", alpha: ", opt.alpha)
    print("data: sample data, CI: ", CI, ", z_dims: ", opt.dim_z, ", data_nums: ", data_nums, "T_scale: ", opt.T_scale, "noise_scale: ", opt.noise_scale)

    repeat_times = opt.repeat_times
    if CI == True:
        # type I error samples:
        print("type I error:")
        print("final--->", "alpha5_", method[0], ": ", med_alpha_all / repeat_times)
        print("final--->", "alpha5_", method[1], ": ", selz_alpha_all / repeat_times)
        print("final--->", "alpha5_", method[2], ": ", sely_alpha_all / repeat_times)
        print("final--->", "alpha5_", method[3], ": ", power_alpha_all / repeat_times)
        print("time: (", round(np.mean(time_list), 3), "+", round(np.var(time_list), 4), ")")
    else:
        # type II error samples:
        print("type II error:")
        print("final--->", "alpha5_", method[0], ": ", 1 - med_alpha_all / repeat_times)
        print("final--->", "alpha5_", method[1], ": ", 1 - selz_alpha_all / repeat_times)
        print("final--->", "alpha5_", method[2], ": ", 1 - sely_alpha_all / repeat_times)
        print("final--->", "alpha5_", method[3], ": ", 1 - power_alpha_all / repeat_times)
        print("time: (", round(np.mean(time_list), 3), "+", round(np.var(time_list), 4), ")")
    print("------------------------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="parameter setting")

    # hypothesis testing and data parameters
    parser.add_argument("--repeat_times", type=int, default=1000, help="repeat times")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level, default 0.05")
    parser.add_argument("--data_nums", type=int, default=500, help="sample size")
    parser.add_argument("--dim_z", type=int, default=5, help="dimension of z")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="noise scale")
    parser.add_argument("--T_scale", type=float, default=0.5,
                        help="The scale of the shared latent variable T when the observed variables are conditionally independent")

    # KCI parameters
    parser.add_argument("--test_method", type=str, default="chi_square",
                        help="testing method: 'chi_square' for weighted sum of chi square and 'gamma' for Gamma approximimation")
    parser.add_argument("--bandwidth_candidates", default=[1, 0.1, 0.3, 0.5, 0.75, 0.88, 1.25, 1.5, 3, 5, 10],
                        help="The candidate list of kernel bandwidths is defined as multiples of the median, wrapped in a list.")
    parser.add_argument("--print_freq", type=int, default=100, help="print frequency")

    opt = parser.parse_args()

    MainTest(CI=True, z_dim=opt.dim_z, data_nums=opt.data_nums, opt=opt)
    MainTest(CI=False, z_dim=opt.dim_z, data_nums=opt.data_nums, opt=opt)

if __name__ == '__main__':
    main()