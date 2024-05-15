import numpy as np
import matplotlib.pyplot as plt
import time
import jax
import jax.numpy as jnp
import optax
from functools import partial
from tqdm import tqdm
import baselines
from kernels import *
from utils import health_economics_utils
import os
import pwd
import shutil
import argparse
import pickle
from jax.config import config

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)


plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()


def get_config():
    parser = argparse.ArgumentParser(description='Conditional Bayesian Quadrature for Bayesian sensitivity analysis')
    # Args settings
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--baseline_use_variance', action='store_true', default=False)
    args = parser.parse_args()
    return args


def f1(theta, x):
    """
    Compute f1.

    Args:
        theta: (T, )
        x: (T, N, 9)

    Returns:
        f(x, theta): (T, N)
    """
    lamba_ = 1e4
    # lamba_ = 1.0
    # lambda * (Theta_5 * Theta_6 * Theta_7 + Theta_8 * Theta_9 * Theta_10) - (Theta_1 + Theta_2 * Theta_3 * Theta_4)
    # Theta_5 is theta
    # Theta_1 is x[0], Theta_2 is x[1], Theta_3 is x[2], Theta_4 is x[3], Theta_6 is x[4], Theta_7 is x[5], Theta_8 is x[6],
    # Theta_9 is x[7], Theta_10 is x[8],
    return lamba_ * (theta * x[:, :, 4] * x[:, :, 5] + x[:, :, 6] * x[:, :, 7] * x[:, :, 8]) - \
           (x[:, :, 0] + x[:, :, 1] * x[:, :, 2] * x[:, :, 3])


def f2(theta, x):
    """
    Compute f2.

    Args:
        theta: (T, )
        x: (T, N, 9)

    Returns:
        f(x, theta): (T, N)
    """
    lamba_ = 1e4
    # lambda * (Theta_5 * Theta_6 * Theta_7 + Theta_8 * Theta_9 * Theta_10) - (Theta_1 + Theta_2 * Theta_3 * Theta_4)
    # Theta_14 is theta
    # Theta_4 is x[0], Theta_11 is x[1], Theta_12 is x[2], Theta_13 is x[3], Theta_15 is x[4], Theta_16 is x[5],
    # Theta_17 is x[6], Theta_18 is x[7], Theta_19 is x[8]
    return lamba_ * (theta * x[:, :, 4] * x[:, :, 5] + x[:, :, 6] * x[:, :, 7] * x[:, :, 8]) - \
           (x[:, :, 1] + x[:, :, 2] * x[:, :, 3] * x[:, :, 0])


def conditional_distribution(joint_mean, joint_covariance, theta, dimensions_x, dimensions_theta):
    """
    Compute conditional distribution p(x | theta).

    Args:
        joint_mean: (19,)
        joint_covariance: (19, 19)
        theta: (N, len(dimensions_theta))
        dimensions_x: list
        dimensions_theta: list

    Returns:
        mean_x_given_theta: shape (N, len(dimensions_x))
        cov_x_given_theta: shape (N, len(dimensions_x), len(dimensions_x))
    """
    dimensions_x = jnp.array(dimensions_x)
    dimensions_theta = jnp.array(dimensions_theta)

    mean_theta = jnp.take(joint_mean, dimensions_theta)[:, None]
    mean_x = jnp.take(joint_mean, dimensions_x)[:, None]

    # Create a grid of indices from A and B using meshgrid
    cov_ThetaTheta = joint_covariance[jnp.ix_(dimensions_theta, dimensions_theta)]
    cov_XX = joint_covariance[jnp.ix_(dimensions_x, dimensions_x)]
    cov_XTheta = joint_covariance[jnp.ix_(dimensions_x, dimensions_theta)]
    cov_ThetaX = joint_covariance[jnp.ix_(dimensions_theta, dimensions_x)]

    mean_x_given_theta = mean_x + cov_XTheta @ jnp.linalg.inv(cov_ThetaTheta) @ (theta.T - mean_theta)
    cov_x_given_theta = cov_XX - cov_XTheta @ jnp.linalg.inv(cov_ThetaTheta) @ cov_ThetaX
    return mean_x_given_theta.T, cov_x_given_theta


def Bayesian_Monte_Carlo_Matern_vectorized(rng_key, U, X, fX):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    Vectorized over Theta.

    Args:
        rng_key: random number generator
        U: shape (T, N, D)
        X: shape (T, N, D)
        f_X: shape (T, N)

    Returns:
        I_BQ_mean: (T, )
        I_BQ_std: (T, )
    """
    scale = 1000
    fX_standardized = fX / scale
    vmap_func = jax.vmap(Bayesian_Monte_Carlo_Matern, in_axes=(None, 0, 0, 0))
    I_BQ_mean, I_BQ_std = vmap_func(rng_key, U, X, fX_standardized)
    return I_BQ_mean * scale, I_BQ_std


def Bayesian_Monte_Carlo_Matern(rng_key, u, x, fx):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    Not vectorized over Theta.

    Args:
        rng_key: random number generator
        u: shape (N, D)
        x: shape (N, D)
        fx: shape (N, )
        
    Returns:
        I_BQ_mean: float
        I_BQ_std: float
    """
    N, D = x.shape[0], x.shape[1]
    eps = 1e-6

    K_no_scale = jnp.zeros([N, N])
    phi_no_scale = jnp.zeros([N, 1])

    l_array = jnp.array([1.0] * 9)
    for i in range(D):
        l = l_array[i]
        u_i = u[:, i][:, None]
        K_no_scale += my_Matern(u_i, u_i, l)
        phi_no_scale += kme_Matern_Gaussian(l, u_i)

    A = fx.T @ K_no_scale @ fx / N
    K = A * K_no_scale
    phi = A * phi_no_scale

    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    varphi = phi.mean()

    CBQ_mean = phi.T @ K_inv @ fx
    CBQ_std = jnp.sqrt(jnp.abs(varphi - phi.T @ K_inv @ phi))

    CBQ_mean = CBQ_mean.squeeze()
    CBQ_std = CBQ_std.squeeze()
    pause = True
    return CBQ_mean, CBQ_std


def GP(I_mean, I_std, Theta, Theta_test, eps):
    """
    Second stage of CBQ, computes the posterior mean and variance of I(Theta_test).
    The kernel hyperparameters are selected by minimizing the negative log-likelihood (NLL).

    Args:
        rng_key: random number generator
        I_mean: (T, )
        I_std: (T, )
        Theta: (T, D)
        Theta_test: (T_test, D)
        eps: float
        kernel_fn: Matern or RBF

    Returns:
        mu_Theta_test: (T_test, )
        std_Theta_test: (T_test, )
    """
    T, D = Theta.shape[0], Theta.shape[1]
    l_array = jnp.array([0.3, 1.0, 2.0, 3.0])

    A_array = 0 * l_array
    nll_array = jnp.zeros([len(l_array), 1])

    for i, l in enumerate(l_array):
        K_no_scale = my_Matern(Theta, Theta, l)
        A = I_mean.T @ K_no_scale @ I_mean / T
        A_array = A_array.at[i].set(A.squeeze())
        K = A * my_Matern(Theta, Theta, l) + eps * jnp.eye(T) + jnp.diag(I_std ** 2)
        K_inv = jnp.linalg.inv(K)
        nll = -(-0.5 * I_mean.T @ K_inv @ I_mean - 0.5 * jnp.log(jnp.linalg.det(K) + 1e-6)) / T
        nll_array = nll_array.at[i].set(nll.squeeze())

    l = l_array[jnp.argmin(nll_array)]
    A = A_array[jnp.argmin(nll_array)]

    K_train_train = A * my_Matern(Theta, Theta, l) + eps * jnp.eye(T) + jnp.diag(I_std ** 2)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = A * my_Matern(Theta_test, Theta, l)
    K_test_test = A * my_Matern(Theta_test, Theta_test, l) + eps * jnp.eye(Theta_test.shape[0])

    mu_Theta_test = K_test_train @ K_train_train_inv @ I_mean
    var_Theta_test = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    var_Theta_test = jnp.abs(var_Theta_test)
    std_Theta_test = jnp.sqrt(var_Theta_test)
    pause = True
    return mu_Theta_test, std_Theta_test


def MOBQ(rng_key, u, theta, theta_test, fx):
    """
    Multi-output Bayesian Quadrature

    Args:
        rng_key: random number generator
        u: shape (N, D)
        theta: shape (N, 1)
        theta_test: shape (N_test, 1)
        fx: shape (N, )
    """
    N, D = u.shape[0], u.shape[1]
    eps = 1e-6

    K_U_u_u = jnp.zeros([N, N])
    kme_U = jnp.zeros([N, 1])

    l_array = jnp.array([1.0] * 9)
    for i in range(D):
        l = l_array[i]
        u_i = u[:, i][:, None]
        K_U_u_u += my_Matern(u_i, u_i, l)
        kme_U += kme_Matern_Gaussian(l, u_i)

    K_Theta_theta_theta = my_Matern(theta, theta, l)
    K_Theta_theta_test_theta = my_Matern(theta_test, theta, l)
    MOBQ_mean = (kme_U.T * K_Theta_theta_test_theta) @ jnp.linalg.inv(K_U_u_u * K_Theta_theta_theta + eps * jnp.eye(N)) @ fx
    return MOBQ_mean


def main(args):
    seed = args.seed
    rng_key = jax.random.PRNGKey(seed)
    ThetaX_mean = jnp.array([1000., 0.1, 5.2, 400., 0.7,
                         0.3, 3.0, 0.25, -0.1, 0.5,
                         1500, 0.08, 6.1, 0.8, 0.3,
                         3.0, 0.2, -0.1, 0.5])
    ThetaX_sigma = jnp.array([1.0, 0.02, 1.0, 200, 0.1,
                          0.1, 0.5, 0.1, 0.02, 0.2,
                          1.0, 0.02, 1.0, 0.1, 0.05,
                          1.0, 0.05, 0.02, 0.2])
    ThetaX_sigma = jnp.diag(ThetaX_sigma ** 2)
    ThetaX_sigma = ThetaX_sigma.at[4, 6].set(0.6 * jnp.sqrt(ThetaX_sigma[4, 4]) * jnp.sqrt(ThetaX_sigma[6, 6]))
    ThetaX_sigma = ThetaX_sigma.at[6, 4].set(0.6 * jnp.sqrt(ThetaX_sigma[4, 4]) * jnp.sqrt(ThetaX_sigma[6, 6]))
    ThetaX_sigma = ThetaX_sigma.at[4, 13].set(0.6 * jnp.sqrt(ThetaX_sigma[4, 4]) * jnp.sqrt(ThetaX_sigma[13, 13]))
    ThetaX_sigma = ThetaX_sigma.at[13, 4].set(0.6 * jnp.sqrt(ThetaX_sigma[4, 4]) * jnp.sqrt(ThetaX_sigma[13, 13]))
    ThetaX_sigma = ThetaX_sigma.at[4, 15].set(0.6 * jnp.sqrt(ThetaX_sigma[4, 4]) * jnp.sqrt(ThetaX_sigma[15, 15]))
    ThetaX_sigma = ThetaX_sigma.at[15, 4].set(0.6 * jnp.sqrt(ThetaX_sigma[4, 4]) * jnp.sqrt(ThetaX_sigma[15, 15]))
    ThetaX_sigma = ThetaX_sigma.at[6, 13].set(0.6 * jnp.sqrt(ThetaX_sigma[6, 6]) * jnp.sqrt(ThetaX_sigma[13, 13]))
    ThetaX_sigma = ThetaX_sigma.at[13, 6].set(0.6 * jnp.sqrt(ThetaX_sigma[6, 6]) * jnp.sqrt(ThetaX_sigma[13, 13]))
    ThetaX_sigma = ThetaX_sigma.at[6, 15].set(0.6 * jnp.sqrt(ThetaX_sigma[6, 6]) * jnp.sqrt(ThetaX_sigma[15, 15]))
    ThetaX_sigma = ThetaX_sigma.at[15, 6].set(0.6 * jnp.sqrt(ThetaX_sigma[6, 6]) * jnp.sqrt(ThetaX_sigma[15, 15]))
    ThetaX_sigma = ThetaX_sigma.at[13, 15].set(0.6 * jnp.sqrt(ThetaX_sigma[13, 13]) * jnp.sqrt(ThetaX_sigma[15, 15]))
    ThetaX_sigma = ThetaX_sigma.at[15, 13].set(0.6 * jnp.sqrt(ThetaX_sigma[13, 13]) * jnp.sqrt(ThetaX_sigma[15, 15]))

    # Theta is index (4, 13), X is index (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18)
    Theta_mean = jnp.array([0.7, 0.8])
    Theta_sigma = jnp.array([[0.01, 0.01 * 0.6], [0.01 * 0.6, 0.01]])
    X_mean = jnp.array([1000., 0.1, 5.2, 400.,
                        0.3, 3.0, 0.25, -0.1, 0.5,
                        1500, 0.08, 6.1, 0.3,
                        3.0, 0.2, -0.1, 0.5])
    X_sigma = jnp.array([1.0, 0.02, 1.0, 200,
                         0.1, 0.5, 0.1, 0.02, 0.2,
                         1.0, 0.02, 1.0, 0.05,
                         1.0, 0.05, 0.02, 0.2])
    f1_cond_dist_fn = partial(conditional_distribution, joint_mean=ThetaX_mean, joint_covariance=ThetaX_sigma,
                              dimensions_theta=[4], dimensions_x=[0, 1, 2, 3, 5, 6, 7, 8, 9])
    f2_cond_dist_fn = partial(conditional_distribution, joint_mean=ThetaX_mean, joint_covariance=ThetaX_sigma,
                              dimensions_theta=[13], dimensions_x=[3, 10, 11, 12, 14, 15, 16, 17, 18])

    # ======================================== Code to generate test points ========================================
    T_test = 100
    large_sample_size = 10000
    rng_key, _ = jax.random.split(rng_key)
    Theta_test = jax.random.multivariate_normal(rng_key, Theta_mean, Theta_sigma, shape=(T_test,))
    Theta1_test = Theta_test[:, 0][:, None]
    Theta2_test = Theta_test[:, 1][:, None]
    f1_p_X_Theta_mean_test, f1_p_X_Theta_sigma_test = f1_cond_dist_fn(theta=Theta1_test)
    f2_p_X_Theta_mean_test, f2_p_X_Theta_sigma_test = f2_cond_dist_fn(theta=Theta2_test)

    X1_test = jnp.zeros([T_test, large_sample_size, 9]) + 0.0
    u1_test = jnp.zeros([T_test, large_sample_size, 9]) + 0.0
    X2_test = jnp.zeros([T_test, large_sample_size, 9]) + 0.0
    u2_test = jnp.zeros([T_test, large_sample_size, 9]) + 0.0

    # Use 
    for i in tqdm(range(T_test)):
        rng_key, _ = jax.random.split(rng_key)
        u1_temp = jax.random.multivariate_normal(rng_key,
                                                 mean=jnp.zeros_like(f1_p_X_Theta_mean_test[i, :]),
                                                 cov=jnp.eye(f1_p_X_Theta_sigma_test.shape[0]),
                                                 shape=(large_sample_size,))
        L1 = jnp.linalg.cholesky(f1_p_X_Theta_sigma_test)
        X1_temp = f1_p_X_Theta_mean_test[i, :] + jnp.matmul(L1, u1_temp.T).T
        # X1_temp = jax.random.multivariate_normal(rng_key, f1_p_X_Theta_mean[i, :], f1_p_X_Theta_sigma, shape=(N,))
        X1_test = X1_test.at[i, :, :].set(X1_temp)
        u1_test = u1_test.at[i, :, :].set(u1_temp)

        rng_key, _ = jax.random.split(rng_key)
        u2_temp = jax.random.multivariate_normal(rng_key,
                                                 mean=jnp.zeros_like(f2_p_X_Theta_mean_test[i, :]),
                                                 cov=jnp.eye(f2_p_X_Theta_sigma_test.shape[0]),
                                                 shape=(large_sample_size,))
        L2 = jnp.linalg.cholesky(f2_p_X_Theta_sigma_test)
        X2_temp = f2_p_X_Theta_mean_test[i, :] + jnp.matmul(L2, u2_temp.T).T
        X2_test = X2_test.at[i, :, :].set(X2_temp)
        u2_test = u2_test.at[i, :, :].set(u2_temp)
    f1_X_test = f1(Theta1_test, X1_test)
    f2_X_test = f2(Theta2_test, X2_test)
    ground_truth_1 = f1_X_test.mean(1)
    ground_truth_2 = f2_X_test.mean(1)
    # ======================================== Code to generate test points Ends ========================================

    # T_array = jnp.array([10, 20, 30])
    T_array = jnp.array([10, 30, 50])
    # T_array = jnp.concatenate((jnp.array([3, 5]), jnp.arange(10, 150, 10)))
    #·
    # N_array = jnp.array([10, 30])
    # N_array = jnp.array([10, 30, 50, 100])
    N_array = jnp.arange(50, 200, 10)

    for T in T_array:
        rng_key, _ = jax.random.split(rng_key)
        Theta1 = jnp.linspace(Theta1_test.min(), Theta1_test.max(), T)[:, None]
        Theta2 = jnp.linspace(Theta2_test.min(), Theta2_test.max(), T)[:, None]
        for N in tqdm(N_array):
            # ======================================== Collecting samples and function evaluations ========================================

            f1_p_X_Theta_mean, f1_p_X_Theta_std = f1_cond_dist_fn(theta=Theta1)
            f2_p_X_Theta_mean, f2_p_X_Theta_std = f2_cond_dist_fn(theta=Theta2)
            # X1, X2 shape is (T, N, 9)
            X1 = jnp.zeros([T, N, 9]) + 0.0
            U1 = jnp.zeros([T, N, 9]) + 0.0
            X2 = jnp.zeros([T, N, 9]) + 0.0
            U2 = jnp.zeros([T, N, 9]) + 0.0
            for i in range(T):
                rng_key, _ = jax.random.split(rng_key)
                u1_temp = jax.random.multivariate_normal(rng_key,
                                                         mean=jnp.zeros_like(f1_p_X_Theta_mean[i, :]),
                                                         cov=jnp.eye(f1_p_X_Theta_std.shape[0]),
                                                         shape=(N,))
                L1 = jnp.linalg.cholesky(f1_p_X_Theta_std)
                X1_temp = f1_p_X_Theta_mean[i, :] + jnp.matmul(L1, u1_temp.T).T
                X1 = X1.at[i, :, :].set(X1_temp)
                U1 = U1.at[i, :, :].set(u1_temp)

                rng_key, _ = jax.random.split(rng_key)
                u2_temp = jax.random.multivariate_normal(rng_key,
                                                         mean=jnp.zeros_like(f2_p_X_Theta_mean[i, :]),
                                                         cov=jnp.eye(f2_p_X_Theta_std.shape[0]),
                                                         shape=(N,))
                L2 = jnp.linalg.cholesky(f2_p_X_Theta_std)
                X2_temp = f2_p_X_Theta_mean[i, :] + jnp.matmul(L2, u2_temp.T).T
                X2 = X2.at[i, :, :].set(X2_temp)
                U2 = U2.at[i, :, :].set(u2_temp)
                # ======================================== Collecting samples and function evaluations Ends ========================================

            # f_X shape is (T, N)
            f1_X = f1(Theta1, X1)
            f2_X = f2(Theta2, X2)

            # ======================================== Code for f1 Starts ========================================
            # ======================================== Debug code ========================================
            # for i in range(T):
            #     rng_key, _ = jax.random.split(rng_key)
            #     X1_i = X1[i, :, :]
            #     f1_X_i = f1_X[i, :]
            #     u1_i = u1[i, :, :]
            #     scale = 1000.0
            #     f1_X_i_standardized = f1_X_i / scale
            #     I1_BQ_mean, I1_BQ_std = Bayesian_Monte_Carlo_Matern(rng_key, u1_i, X1_i, f1_X_i_standardized)
            #     I1_BQ_mean = I1_BQ_mean * scale
            #     f1_mc_mean = f1_X_i.mean()
            #     I1_BQ_std *= 10

            #     I1_BQ_mean_array = I1_BQ_mean_array.at[i].set(I1_BQ_mean)
            #     I1_BQ_std_array = I1_BQ_std_array.at[i].set(I1_BQ_std)
            #     I1_MC_mean_array = I1_MC_mean_array.at[i].set(f1_mc_mean)

            #     
            #     rng_key, _ = jax.random.split(rng_key)
            #     X_temp_large = jax.random.multivariate_normal(rng_key, f1_p_X_Theta_mean[i, :], f1_p_X_Theta_sigma,
            #                                                   shape=(large_sample_size,))
            #     f1_X_large = f1(Theta1[i, :][None, :], X_temp_large[None, :])
            #     f1_mc_mean_large = f1_X_large.mean()
            #     print====================
            #     print('Large sample MC', f1_mc_mean_large)
            #     print(f'MC with {N} number of X', f1_mc_mean)
            #     print(f'CBQ with {N} number of X', I1_BQ_mean)
            #     print(f'CBQ uncertainty {I1_BQ_std}')
            #     print(f====================
            #     pause = True
            #     ======================================== Debug code ========================================

            # ======================================== CBQ ========================================
            time0 = time.time()
            I1_BQ_mean, I1_BQ_std = Bayesian_Monte_Carlo_Matern_vectorized(rng_key, U1, X1, f1_X)
            CBQ_mean_1, CBQ_std_1 = GP(I1_BQ_mean, I1_BQ_std, Theta1, Theta1_test, eps=I1_BQ_std.mean())
            time_CBQ = time.time() - time0
            # ======================================== CBQ ========================================

            # ======================================== LSMC ========================================
            time0 = time.time()
            if args.baseline_use_variance:
                LSMC_mean_1, LSMC_std_1 = baselines.polynomial(Theta1, X1, f1_X, Theta1_test, baseline_use_variance=True)
            else:
                LSMC_mean_1, LSMC_std_1 = baselines.polynomial(Theta1, X1, f1_X, Theta1_test, baseline_use_variance=False)
            time_LSMC = time.time() - time0
            # ======================================== LSMC ========================================

            # ======================================== KMS ========================================
            time0 = time.time()
            I1_MC_mean = f1_X.mean(1)
            I1_MC_std = f1_X.std(1)
            if args.baseline_use_variance:
                KMS_mean_1, KMS_std_1 = baselines.kernel_mean_shrinkage(rng_key, I1_MC_mean, I1_MC_std, Theta1, Theta1_test, eps=0., kernel_fn=my_RBF)
            else:
                KMS_mean_1, KMS_std_1 = baselines.kernel_mean_shrinkage(rng_key, I1_MC_mean, None, Theta1, Theta1_test, eps=0., kernel_fn=my_RBF)
            time_KMS = time.time() - time0
            # ======================================== KMS ========================================

            # ======================================== Code for f1 Ends ========================================

            # ======================================== Code for f2 Starts ========================================
            
            # ======================================== CBQ ========================================
            I2_BQ_mean, I2_BQ_std = Bayesian_Monte_Carlo_Matern_vectorized(rng_key, U2, X2, f2_X)
            CBQ_mean_2, CBQ_std_2 = GP(I2_BQ_mean, I2_BQ_std, Theta2, Theta2_test, eps=I2_BQ_std.mean())
            # ======================================== CBQ ========================================

            # ======================================== LSMC ========================================
            if args.baseline_use_variance:
                LSMC_mean_2, LSMC_std_2 = baselines.polynomial(Theta2, X2, f2_X, Theta2_test, baseline_use_variance=True)
            else:
                LSMC_mean_2, LSMC_std_2 = baselines.polynomial(Theta2, X2, f2_X, Theta2_test, baseline_use_variance=False)
            # ======================================== LSMC ========================================

            # ======================================== KMS ========================================
            I2_MC_mean = f2_X.mean(1)
            I2_MC_std = f2_X.std(1)
            if args.baseline_use_variance:
                KMS_mean_2, KMS_std_2 = baselines.kernel_mean_shrinkage(rng_key, I2_MC_mean, I2_MC_std, Theta2, Theta2_test, eps=0., kernel_fn=my_RBF)
            else:
                KMS_mean_2, KMS_std_2 = baselines.kernel_mean_shrinkage(rng_key, I2_MC_mean, None, Theta2, Theta2_test, eps=0., kernel_fn=my_RBF)
            # ======================================== KMS ========================================

            # ======================================== Code for f2 Ends ========================================

            calibration_1 = health_economics_utils.calibrate(ground_truth_1, CBQ_mean_1, jnp.diag(CBQ_std_1))
            calibration_2 = health_economics_utils.calibrate(ground_truth_2, CBQ_mean_2, jnp.diag(CBQ_std_2))

            true_value = jnp.maximum(ground_truth_1, ground_truth_2)
            CBQ_value = jnp.maximum(CBQ_mean_1, CBQ_mean_2)
            KMS_value = jnp.maximum(KMS_mean_1, KMS_mean_2)
            LSMC_value = jnp.maximum(LSMC_mean_1, LSMC_mean_2)

            rmse_CBQ = jnp.sqrt(jnp.mean((CBQ_value - true_value) ** 2))
            rmse_KMS = jnp.sqrt(jnp.mean((KMS_value - true_value) ** 2))
            rmse_LSMC = jnp.sqrt(jnp.mean((LSMC_value - true_value) ** 2))

            health_economics_utils.save(args, T, N, rmse_CBQ, rmse_KMS, rmse_LSMC, 
                                        time_CBQ, time_KMS, time_LSMC, 
                                        calibration_1, calibration_2)
            # ======================================== Debug code ========================================
            # plt.figure()
            # Theta1_test_ = Theta1_test.squeeze()
            # ind = Theta1_test_.argsort()
            # plt.plot(Theta1_test_[ind], ground_truth_1[ind], label='Ground truth')
            # plt.plot(Theta1_test_[ind], CBQ_mean_1[ind], label='CBQ')
            # plt.plot(Theta1_test_[ind], KMS_mean_1[ind], label='KMS')
            # plt.scatter(Theta1.squeeze(), I1_BQ_mean.squeeze())
            # plt.plot(Theta1_test_[ind], LSMC_mean_1[ind], label='LSMC')
            # plt.legend()
            # plt.savefig(f"{args.save_path}/debug1.png")
            
            # plt.figure()
            # Theta2_test_ = Theta2_test.squeeze()
            # ind = Theta2_test_.argsort()
            # plt.plot(Theta2_test_[ind], ground_truth_2[ind], label='Ground truth')
            # plt.plot(Theta2_test_[ind], CBQ_mean_2[ind], label='CBQ')
            # plt.plot(Theta2_test_[ind], KMS_mean_2[ind], label='KMS')
            # plt.scatter(Theta2.squeeze(), I2_BQ_mean.squeeze())
            # plt.plot(Theta2_test_[ind], LSMC_mean_2[ind], label='LSMC')
            # plt.legend()
            # plt.savefig(f"{args.save_path}/debug2.png")
            # pause = True
            # ======================================== Debug code ========================================



def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/decision/'
    args.save_path += f"seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    return args


if __name__ == '__main__':
    args = get_config()
    create_dir(args)
    print(f'Device is {jax.devices()}')
    print(f'Seed is {args.seed}')
    main(args)
    save_path = args.save_path
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    if os.path.exists(f"{save_path}__complete"):
        print(f"Removing old results at {save_path}__complete")
        shutil.rmtree(f"{save_path}__complete")
    os.rename(save_path, f"{save_path}__complete")
    print("\n------------------- DONE -------------------\n")
