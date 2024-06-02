#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import warnings
warnings.filterwarnings('ignore')

import tqdm

import numpy as np
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator as RGI

from scipy.sparse import diags,csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, gmres
from krypy.linsys import LinearSystem
from krypy.linsys import Minres as Minres_kry
from krypy.linsys import Gmres as Gmres_kry
from krypy.recycling import RecyclingMinres, RecyclingGmres


import torch
import matplotlib.pyplot as plt
import math


from scipy.fftpack import idct

import numpy.polynomial.chebyshev as cheb 
from ipdb import set_trace
import time
from torch.multiprocessing import Pool
# from utils import liner
# from torch.sparse import spdiags



def GRF2(alpha, tau, s):
    # Random variables in KL expansion
    xi = np.random.randn(s, s)

    # Define the (square root of) eigenvalues of the covariance operator
    K1, K2 = np.meshgrid(np.arange(s), np.arange(s))
    coef = (tau**(alpha-1) * (np.pi**2 * (K1**2 + K2**2) + tau**2)**(-alpha/2))

    # Construct the KL coefficients
    L = s * coef * xi
    L[0, 0] = 0

    # Inverse Discrete Cosine Transform (IDCT)
    U = idct(idct(L, norm='ortho').T, norm='ortho').T

    return U

class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real

### 这个函数实现的darcy flow考虑了a的变化
def solve_gwf(coef, F):
    K = coef.shape[0]

    X1, Y1 = np.meshgrid(np.arange(1/(2*K), 1, 1/K), np.arange(1/(2*K), 1, 1/K))
    X2, Y2 = np.meshgrid(np.arange(0, 1, 1/(K-1)), np.arange(0, 1, 1/(K-1)))

    # interp_func = RGI((X1,Y1),coef.T,method='linear')
    interp_func = interp2d(X1, Y1, coef, kind='linear')
    coef = interp_func(X2[0,:], Y2[:,0])

    # interp_func_F = interp2d(X1, Y1, F, kind='linear')
    interp_func_F = RGI((X1,Y1),F.T,method='linear')
    F = interp_func_F(X2[0,:], Y2[:,0])

    F = F[1:K-1, 1:K-1]

    d = [[None]* (K-2) for _ in range(K-2)]

    for j in range(1, K-1):
        diag_values = np.array([
            -0.5 * (coef[1:K-2, j] + coef[2:K-1, j]),
            0.5 * (coef[0:K-2, j] + coef[1:K-1, j]) + 0.5 * (coef[2:K, j] + coef[1:K-1, j]) + \
            0.5 * (coef[1:K-1, j-1] + coef[1:K-1, j]) + 0.5 * (coef[1:K-1, j+1] + coef[1:K-1, j]),
            np.concatenate(([0], -0.5 * (coef[1:K-2, j] + coef[2:K-1, j])))
        ])

        d[j-1][j-1] = diags(diag_values, [-1, 0, 1], (K-2, K-2))

        if j != K-2:
            off_diag = diags(-0.5 * (coef[1:K-1, j] + coef[1:K-1, j+1]), 0, (K-2, K-2))
            d[j-1][j] = off_diag
            d[j][j-1] = off_diag
        print(j)
    A = np.bmat(d) * (K-1)**2
    P = np.zeros((K, K))
    P[1:K-1, 1:K-1] = np.reshape(np.linalg.solve(A.todense(), F.ravel()), (K-2, K-2))

    interp_func = interp2d(X2, Y2, P, kind='cubic')
    P = interp_func(X1[0,:], Y1[:,0])

    return P.T


def build_darcy2(coef, F):
    K = coef.shape[0]
    s = K - 2
    
    F = F[1:K-1, 1:K-1]
    zeros = np.zeros((K-2,K-2))
    # sparse_zeros = csr_matrix(([0.0], ([1,1])),shape=(K-2,K-2), dtype=np.float32)

    d = [[zeros]* (K-2) for _ in range(K-2)]

    diag_list = []
    off_diag_list = []
    for j in range(1, K-1):

        diag_values = np.array([
            np.concatenate((-0.5 * (coef[1:K-2, j] + coef[2:K-1, j]),[0])),
            0.5 * (coef[0:K-2, j] + coef[1:K-1, j]) + 0.5 * (coef[2:K, j] + coef[1:K-1, j]) + \
            0.5 * (coef[1:K-1, j-1] + coef[1:K-1, j]) + 0.5 * (coef[1:K-1, j+1] + coef[1:K-1, j]),
            np.concatenate((-0.5 * (coef[1:K-2, j] + coef[2:K-1, j]),[0]))
        ])
        diag_list.append(diag_values)

        if j != K-2:
            off_diag = -0.5 * (coef[1:K-1, j] + coef[1:K-1, j+1])
            off_diag_list.append(off_diag)

    diag_output = np.concatenate(diag_list,axis=1)
    off_diag_output = np.concatenate(off_diag_list,axis=0)
    A = (diags(diag_output,[-1,0,1],(s**2,s**2)) + diags((off_diag_output,off_diag_output),[-(K-2),(K-2)],(s**2,s**2))) * (K-1)**2

    return A, F.flatten()

### 这个函数的实现把系数a提了出来
def solve_darcy1(coef, f):

    # 定义问题参数
    s = coef.shape[0] - 2  # 网格尺寸
    h = 1.0 / (s - 1)  # 网格步长

    f = f[1:-1,1:-1]
    coef_pos = coef[1:-1,2:]
    coef_pos[:,-1] = 0
    coef_neg = coef[1:-1,:-2]
    coef_neg[:,0] = 0
    # 计算系数矩阵
    # coef_flat = coef.flatten()
    diagonals = [-coef_neg.flatten()[1:], -coef[:-2,1:-1].flatten()[s-2:], 4*coef[1:-1,1:-1].flatten(), -coef[2:,1:-1].flatten()[:-(s-2)], -coef_pos.flatten()[:-1]]
    offsets = [-1, -s, 0, s, 1]
    A = diags(diagonals, offsets, shape=(s*s, s*s))

    ## removes

    # 创建向量b
    b = h ** 2 * f.flatten()

    # 使用scipy的稀疏线性系统求解器求解
    # u, exitcode = gmres(A, b)
    u = spsolve(A, b)
    print(np.allclose(A.dot(u), b))

    # 将解向量重新整形为网格形式
    u = u.reshape((s, s))

    return u


def build_darcy(coef, f):
    # set_trace()

    s = coef.shape[0] - 2  # 网格尺寸
    h = 1.0 / (s - 1)  # 网格步长

    f = f[1:-1, 1:-1]
    coef_pos = coef[1:-1, 2:]
    coef_pos[:, -1] = 0
    coef_neg = coef[1:-1, :-2]
    coef_neg[:, 0] = 0
    # 计算系数矩阵
    # coef_flat = coef.flatten()

    diagonals = [-coef_neg.flatten()[1:], -coef[:-2, 1:-1].flatten()[s - 2:], 4 * coef[1:-1, 1:-1].flatten(),
                 -coef[2:, 1:-1].flatten()[:-(s - 2)], -coef_pos.flatten()[:-1]]
    offsets = [-1, -s, 0, s, 1]
    # set_trace()

    A = diags(diagonals, offsets, shape=(s * s, s * s))


    ## removes

    # 创建向量b
    b = h ** 2 * f.flatten()
    return A, b


def build_darcy3(coef, F):
    K = coef.shape[0]
    # coef = coef.numpy()
    s = K - 2

    F = F[1:K-1, 1:K-1]
    zeros = np.zeros((K-2,K-2))

    d = [[zeros]* (K-2) for _ in range(K-2)]

    diag_list = []
    off_diag_list = []
    for j in range(1, K-1):
        diag_values = np.array([
            np.concatenate((-0.5 * (coef[1:K-2, j] + coef[2:K-1, j]),[0])),
            0.5 * (coef[0:K-2, j] + coef[1:K-1, j]) + 0.5 * (coef[2:K, j] + coef[1:K-1, j]) + \
            0.5 * (coef[1:K-1, j-1] + coef[1:K-1, j]) + 0.5 * (coef[1:K-1, j+1] + coef[1:K-1, j]),
            np.concatenate((-0.5 * (coef[1:K-2, j] + coef[2:K-1, j]),[0]))
        ])
        diag_list.append(diag_values)

        if j != K-2:
            off_diag = -0.5 * (coef[1:K-1, j] + coef[1:K-1, j+1])
            off_diag_list.append(off_diag)

    diag_output = np.concatenate(diag_list,axis=1)
    off_diag_output = np.concatenate(off_diag_list,axis=0)
    A = (diags(diag_output,[-1,0,1],(s**2,s**2)) + diags((off_diag_output,off_diag_output),[-(K-2),(K-2)],(s**2,s**2))) * (K-1)**2

    return A, F.flatten()


def Chebyshev_truncation_polynomial(s, long, N):
    # long 是切比雪夫多项式项数，N是生成的矩阵个数
    x, y = np.meshgrid(np.linspace(0, 1, s),np.linspace(0, 1, s))
    result = np.zeros((N, s, s))
    for i in range(N):
        coeff=np.random.normal(0, 1, size=(long,long))
        coeff/= np.linalg.norm(coeff, ord='fro')
        result[i] = cheb.chebval2d(x, y, coeff)
    result = torch.tensor(result,dtype=torch.float32)
    return result


if __name__ == "__main__":
    s = 152
    edge = s-2
    cheby_num = 30

    train_resolution=s

    train_num,test_num=1000,100
    num_examples = train_num + test_num

    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device='cpu')


    x=[]
    y=[]
    F=[]
    # fset = np.exp(Chebyshev_truncation_polynomial(s, cheby_num, num_examples))
    # w = np.exp(Chebyshev_truncation_polynomial(s, cheby_num, num_examples))
    w = np.exp(GRF.sample(num_examples))
    fset = np.exp(GRF.sample(num_examples))
    # rminres = RecyclingGmres(vector_factory='RitzApproxKrylov')
    h_zeros = torch.zeros((1,s),dtype = torch.float32)  
    v_zeros = torch.zeros((s-2,1),dtype = torch.float32)

    pbar = tqdm.tqdm(desc="Generating",total=num_examples,position=0,leave=True)
    
    t_creat=0
    t_solve=0

    for i in range(num_examples):
        pbar.update()
        wi = w[i]
        f = fset[i].numpy()
        x.append(wi.unsqueeze(0))

        # 这里组装方程
        # t1 = time.time()
        A1, b1 = build_darcy2(wi, f)
        # t2 = time.time()

        ori_solver = gmres(A1, b1)[0]

        y_hat = torch.tensor(ori_solver).reshape(edge,edge)
        y_hat = torch.hstack((v_zeros,y_hat,v_zeros))
        y_hat = torch.vstack((h_zeros,y_hat,h_zeros))
        y.append(y_hat.unsqueeze(0))
        f = torch.tensor(f)
        F.append(f.unsqueeze(0))


    pbar.close()
    

    """
    输出格式对齐就ok，其他可以随便修改，具体细节我们可以再讨论
    """

    x_train = x[:train_num]
    x_test = x[train_num:]

    y_train = y[:train_num]
    y_test = y[train_num:]

    f_train = F[:train_num]
    f_test = F[train_num:]
    
    train_data = {"x":torch.cat(x_train,dim=0),
                  "y":torch.cat(y_train,dim=0),
                  "f":torch.cat(f_train,dim=0)}
    
    test_data = {"x":torch.cat(x_test,dim=0),
                  "y":torch.cat(y_test,dim=0),
                  "f":torch.cat(f_test,dim=0)}
    
    torch.save(train_data,f"darcy_train_{train_resolution}_fGRF.pt")

    torch.save(test_data,f"darcy_test_{train_resolution}_fGRF.pt")


