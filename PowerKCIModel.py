from numpy import median, sqrt
from numpy.random import permutation
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats

import numpy as np
import torch.nn
from utils import data_split, reduce_func
from numpy import shape
from numpy.linalg import eigh
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from joblib import Parallel, delayed
import warnings

class power_KCI(object):
    def __init__(self, Z, X, Y, opt, null_samples = 5000, thresh_hold = 1e-5):
        self.null_samples = null_samples # null sample size
        self.thresh = thresh_hold # SVD threshold
        self.test_method = opt.test_method # test method to get p-value: chi_square or gamma

        # data split and convert from np.array to tensor
        (self.z_tr, self.z_te,
         self.x_tr, self.x_te,
         self.y_tr, self.y_te) = data_split(Z, X, Y)
        self.Dz = self.z_tr.shape[-1]

        # data initialization using median heuristic
        self.xlength_init = self.set_median_width(self.x_tr) # bandwidth of KX in statistic, which is trainable
        self.ylength_init = self.set_median_width(self.y_tr) # bandwidth of KY in statistic, which is trainable
        self.zlength_init = self.set_median_width(self.z_tr) # bandwidth of Z in Gaussian process, which is trainable
        self.zlength_med =  self.set_median_width(self.z_tr) # bandwidth of Kz in the stat, which is kept fixed

        # kernel choice
        self.xselecting_weights = opt.bandwidth_candidates
        self.xweights_nums = len(self.xselecting_weights)
        self.yselecting_weights = self.xselecting_weights
        self.yweights_nums = len(self.yselecting_weights)
        self.kz_weight_list = [1] # opt.bandwidth_candidates
        self.kz_weight_nums = len(self.kz_weight_list)

    def set_median_width(self, X):
        """Compute median pairwise distance as kernel bandwidth."""
        n = shape(X)[0]
        if n > 1000:
            X = X[permutation(n)[:1000], :]
        dists = squareform(pdist(X, 'euclidean'))
        median_dist = median(dists[dists > 0])
        width = median_dist
        return width

    def set_empirical_width(self, X):
        """Heuristic rule: bandwidth based on sample size. (from original KCI matlab implementation)"""
        n = np.shape(X)[0]
        if n < 200:
            width = 1.2
        elif n < 1200:
            width = 0.7
        else:
            width = 0.4
        length = width / np.sqrt(X.shape[1])
        return length

    def get_matrix(self):
        """Compute centered kernel matrices for all candidate bandwidths: KX and KY."""
        K_list = []
        for i in range(self.xweights_nums):
            xlength = sqrt(self.xselecting_weights[i])*self.xlength_init
            Kx_tr = self.cal_kernel(self.x_tr, xlength)
            Kxc_tr = self.kernel_centering(Kx_tr)
            Kx_te = self.cal_kernel(self.x_te, xlength)
            Kxc_te = self.kernel_centering(Kx_te)
            K_list.append([("x", i), Kxc_tr, Kxc_te])

        for j in range(self.yweights_nums):
            ylength = sqrt(self.yselecting_weights[j])*self.ylength_init
            Ky_tr = self.cal_kernel(self.y_tr, ylength)
            Kyc_tr = self.kernel_centering(Ky_tr)
            Ky_te = self.cal_kernel(self.y_te, ylength)
            Kyc_te = self.kernel_centering(Ky_te)
            K_list.append([("y", j), Kyc_tr, Kyc_te])
        return K_list

    def residual(self, gp, z, Kcx):
        """Compute residual kernel matrix using Gaussian Process regression on Z."""
        # np parameters
        noise_scale = np.exp(gp.kernel_.theta[-1])
        Kz = gp.kernel_.k1(z)
        n = shape(Kz)[0]
        # torch parameters
        noise_scale_t = torch.tensor(noise_scale)
        Kzx_t = torch.from_numpy(Kz)
        try:
            Rz = noise_scale_t * torch.linalg.inv(Kzx_t + noise_scale_t * torch.eye(n))
        except:
            Rz = noise_scale_t * torch.linalg.pinv(Kzx_t + noise_scale_t * torch.eye(n))

        Kxc_t = Kcx.to(torch.float64)
        KR = Rz.matmul(Kxc_t.matmul(Rz))
        return KR

    def get_residual_matrix(self, K_one):
        """Fit GP and compute residual matrices for one kernel candidate."""
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        index = K_one[0]
        Kc_tr = K_one[1]
        Kc_te = K_one[2]
        gpx = self.gp(self.zlength_init)
        phi_x = reduce_func(Kc_tr, self.thresh)
        phi_x = torch.from_numpy(phi_x)
        gpx.fit(self.z_tr.numpy(), phi_x)

        KR_tr = self.residual(gpx, self.z_tr, Kc_tr)
        KR_te = self.residual(gpx, self.z_te, Kc_te)
        return [index, KR_tr.detach(), KR_te.detach()]

    # kernel selection function
    def select_kernels(self):
        """Select kernel bandwidths that maximize the estimated power statistic ."""
        Kx_Ky_list = self.get_matrix()
        KR_one = Parallel(n_jobs=-1)(
            delayed(self.get_residual_matrix)(K_one) for K_one in Kx_Ky_list
            )

        # precompute residual and Z-kernels
        Kz_tr_list = []
        for ind_z in range(self.kz_weight_nums):
            zlength = sqrt(self.kz_weight_list[ind_z])*self.zlength_med
            Kz_tr = self.kernelz(self.z_tr, zlength).detach()
            Kz_tr_list.append(Kz_tr)

        KRx_tr_list = []
        KRx_te_list = []
        KRy_tr_list = []
        KRy_te_list = []

        assert(len(KR_one) == self.xweights_nums + self.yweights_nums)
        for i in range(len(KR_one)):
            index = KR_one[i][0]
            if index[0] == "x":
                KRx_tr_list.append(KR_one[i][1])
                KRx_te_list.append(KR_one[i][2])
            elif index[0] == "y":
                KRy_tr_list.append(KR_one[i][1])
                KRy_te_list.append(KR_one[i][2])
            else:
                raise ValueError("index not complete")

        # init the best choice with max estiamated power
        max_xyz_power = -100
        best_xyz = (-1, -1, -1)

        max_yz_power = -100
        best_yz = (-1, -1)

        max_z_power = -100
        best_z = -1

        # Search for best combination
        for i in range(self.xweights_nums):
            for j in range(self.yweights_nums):
                KxRes = KRx_tr_list[i]
                KyRes = KRy_tr_list[j]
                for ind_z in range(self.kz_weight_nums):
                    Kz_tr = Kz_tr_list[ind_z]
                    est_J_sel = self.power(KxRes.clone(), KyRes.clone(), Kz_tr)

                    # median combination tracking
                    if self.xselecting_weights[i] == 1. and self.yselecting_weights[j] == 1. and est_J_sel > max_z_power:
                        max_z_power = est_J_sel
                        best_z = ind_z

                    if self.xselecting_weights[i] == 1. and est_J_sel > max_yz_power:
                        max_yz_power = est_J_sel
                        best_yz = (j, ind_z)

                    # selecting the parameters with maximum of power
                    if est_J_sel > max_xyz_power:
                        max_xyz_power = est_J_sel
                        best_xyz = (i, j, ind_z)

        # compute p-values for selected bandwidths
        # median xyz
        Kz_te_med = self.kernelz(self.z_te, self.zlength_med).detach()
        med_pvalue, _ = self.cal_pvalue(KRx_te_list[0].detach(), KRy_te_list[0].detach(), Kz_te_med)

        # median xy with selected z
        Kz_te_bestz = self.kernelz(self.z_te, sqrt(self.kz_weight_list[best_z])*self.zlength_med).detach()
        sel_z_pvalue, _ = self.cal_pvalue(KRx_te_list[0].detach(), KRy_te_list[0].detach(), Kz_te_bestz)

        # median x with selected y and z
        Kz_te_bestyz = self.kernelz(self.z_te, sqrt(self.kz_weight_list[best_yz[1]])*self.zlength_med).detach()
        sel_yz_pvalue, _ = self.cal_pvalue(KRx_te_list[0].detach(), KRy_te_list[best_yz[0]].detach(), Kz_te_bestyz)

        # best
        (i, j, ind_z) = best_xyz
        Kz_te_bestxyz = self.kernelz(self.z_te, sqrt(self.kz_weight_list[ind_z])*self.zlength_med).detach()
        sel_xyz_pvalue, _ = self.cal_pvalue(KRx_te_list[i].detach(), KRy_te_list[j].detach(), Kz_te_bestxyz)
        return sel_xyz_pvalue, sel_yz_pvalue, sel_z_pvalue, med_pvalue


    def cal_pvalue(self, KxR, KyR, Kz):
        """Compute p-value for test statistic using chi-square or gamma approximation."""
        KxR = KxR * Kz
        KyR = self.kernel_centering(KyR)
        test_stat = torch.sum(KxR*KyR).detach().numpy()

        uu_prod, size_u = self.get_uuprod(KxR, KyR)
        if self.test_method == 'gamma':
            k_appr, theta_appr = self.get_kappa(uu_prod)
            pvalue = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)
        elif self.test_method == 'chi_square':
            null_samples = self.null_sample_spectral(uu_prod, size_u, KxR.shape[0])
            pvalue = sum(null_samples > test_stat) / float(self.null_samples)
        else:
            raise NotImplementedError('test method not implemented')
        return pvalue, test_stat

    def power(self, Kx, Ky, Kz):
        """Compute normalized test statistic J (empirical power)."""
        n = Kx.shape[0]
        S = Kx*Ky*Kz
        S = self.diag_zero(S)
        KCIu = torch.sum(S)
        Sj = S.sum(0) / (n-1)
        # sigma1 = 1
        sigma1 = torch.sqrt(torch.sum((Sj - KCIu)**2) / n + 1e-10)
        J = KCIu / sigma1
        return J

    # HSIC form
    # def power(self, Kx, Ky, Kz):
    #     #HSIC
    #     m = Kx.shape[0]
    #
    #     Kx = Kx.float().clone()
    #     Ky = Ky.float().clone()
    #     Kz = Kz.float().clone()
    #
    #     Kx = Kx * Kz
    #
    #     HSICu = self.HSICu_func(Kx, Ky)
    #     R = self.cal_R(Kx, Ky)
    #
    #     var = torch.sqrt((R - HSICu**2)*16/m)
    #     power = HSICu / var
    #
    #     return power.numpy()
    #
    # @staticmethod
    # def HSICu_func(K, L):
    #     m = K.shape[0]
    #     K.fill_diagonal_(0.)
    #     L.fill_diagonal_(0.)
    #     Kxy = torch.matmul(K, L)
    #     TEMP1 = torch.trace(Kxy)
    #     TEMP2 = K.sum()*L.sum() / (m-1) / (m-2)
    #     TEMP3 = 2*Kxy.sum() / (m-2)
    #     HSIC_u = (TEMP1 + TEMP2 - TEMP3) / m / (m-3)
    #     return HSIC_u
    #
    # def cal_R(self, K, L):
    #     m = K.shape[0]
    #     vec_I = torch.ones((m, 1))
    #     K.fill_diagonal_(0.)
    #     L.fill_diagonal_(0.)
    #
    #     Kxy = torch.matmul(K, L)
    #     K1 = K.matmul(vec_I)
    #     L1 = L.matmul(vec_I)
    #
    #     TEMP1 = torch.mul(K, L).matmul(vec_I) * (m-2)**2
    #     TEMP2 = ((torch.trace(Kxy)) - Kxy - L.matmul(K)).matmul(vec_I) * (m-2)
    #     TEMP3 = torch.mul(K1, L1) * m
    #     TEMP4 = (vec_I.T.matmul(L1))*K1
    #     TEMP5 = vec_I.T.matmul(K1)*L1
    #     TEMP6 = vec_I.T.matmul(Kxy).matmul(vec_I)*(vec_I)
    #     h = TEMP1 + TEMP2 - TEMP3 + TEMP4 + TEMP5 - TEMP6
    #     R = h.T.matmul(h)/(4*m)/(m-1)**2/(m-2)**2/(m-3)**2
    #     return R

    def kernelz(self, z, zlength):
        z = z / zlength
        zsq = (z ** 2).sum(dim=1, keepdim=True)
        sqdist = zsq + zsq.T - 2 * z.mm(z.T)
        Kz =  torch.exp(- 0.5 * sqdist)
        return Kz

    def diag_zero(self, K):
        """Set diagonal elements of a matrix to zero."""
        diag_vec = K.diag()
        diag_mat = torch.diag_embed(diag_vec)
        return K - diag_mat


    def cal_kernel(self, z, zlength):
        z = z / zlength
        zsq = (z ** 2).sum(dim=1, keepdim=True)
        sqdist = zsq + zsq.T - 2 * z.mm(z.T)
        Kz =  torch.exp(- 0.5 * sqdist)
        return Kz

    def gp(self, zlength_init):
        """Build Gaussian Process with RBF + WhiteKernel for residual regression."""
        kernelx = (ConstantKernel(1.0, (1e-2, 1e2))
                             * RBF(zlength_init * np.ones(self.Dz), (1e-2, 1e1))
                       + WhiteKernel(0.1, (1e-6, 1e+1)))
        gpx = GaussianProcessRegressor(kernel=kernelx)
        return gpx

    def regression_residual(self, Kx, Kz, epsilon):
        """Compute regression residual matrix using kernel ridge formulation."""
        n = Kx.shape[0]
        try:
            Rz = epsilon*torch.linalg.inv(Kz + epsilon * torch.eye(n))
        except:
            Rz = epsilon*torch.linalg.pinv(Kz + epsilon * torch.eye(n))

        return Rz.matmul(Kx.matmul(Rz))

    @staticmethod
    def kernel_centering(K):
        """Center kernel matrix (double centering)."""
        n = shape(K)[0]
        K_colsums = K.sum(axis=0)
        K_allsum = K_colsums.sum()
        return K - (K_colsums[None, :] + K_colsums[:, None]) / n + (K_allsum / n ** 2)

    def get_uuprod(self, Kx, Ky):
        """Compute spectral product of kernel eigenvectors for null distribution."""
        wx, vx = eigh(0.5 * (Kx + Kx.T))
        wy, vy = eigh(0.5 * (Ky + Ky.T))
        idx = np.argsort(-wx)
        idy = np.argsort(-wy)
        wx = wx[idx]
        vx = vx[:, idx]
        wy = wy[idy]
        vy = vy[:, idy]
        vx = vx[:, wx > np.max(wx) * self.thresh]
        wx = wx[wx > np.max(wx) * self.thresh]
        vy = vy[:, wy > np.max(wy) * self.thresh]
        wy = wy[wy > np.max(wy) * self.thresh]
        vx = vx.dot(np.diag(np.sqrt(wx)))
        vy = vy.dot(np.diag(np.sqrt(wy)))

        T = Kx.shape[0]
        num_eigx = vx.shape[1]
        num_eigy = vy.shape[1]
        size_u = num_eigx * num_eigy
        uu = np.zeros((T, size_u))
        for i in range(0, num_eigx):
            for j in range(0, num_eigy):
                uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

        if size_u > T:
            uu_prod = uu.dot(uu.T)
        else:
            uu_prod = uu.T.dot(uu)
        return uu_prod, size_u


    def get_kappa(self, uu_prod):
        """Approximate null distribution with gamma(k, θ) parameters."""
        mean_appr = np.trace(uu_prod)
        var_appr = 2 * np.trace(uu_prod.dot(uu_prod))
        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr
        return k_appr, theta_appr

    def null_sample_spectral(self, uu_prod, size_u, T):
        """Generate null distribution samples via spectral approximation."""
        from numpy.linalg import eigvalsh

        eig_uu = eigvalsh(uu_prod)
        eig_uu = -np.sort(-eig_uu)
        eig_uu = eig_uu[0:np.min((T, size_u))]
        eig_uu = eig_uu[eig_uu > np.max(eig_uu) * self.thresh]

        f_rand = np.random.chisquare(1, (eig_uu.shape[0], self.null_samples))
        null_dstr = eig_uu.T.dot(f_rand)
        return null_dstr