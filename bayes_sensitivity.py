import jax
import jax.numpy as jnp
import os
import shutil
import pwd
import jax.scipy
import jax.scipy.stats
import matplotlib.pyplot as plt
import argparse
import baselines
from tqdm import tqdm
from kernels import *
from utils import sensitivity_utils
import time
import optax
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
    parser.add_argument('--dim', type=int)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--fn', type=str, default='f3')
    parser.add_argument('--qmc', action='store_true', default=False)
    parser.add_argument('--kernel_x', type=str)
    parser.add_argument('--kernel_theta', type=str)
    parser.add_argument('--baseline_use_variance', action='store_true', default=False)
    parser.add_argument('--nystrom', action='store_true', default=False)
    args = parser.parse_args()
    return args


def generate_data(rng_key, D, N, noise):
    """
    Generates data for the Bayesian linear regression

    Args:
        rng_key: random number generator
        D: int
        N: int
        noise: float
    Returns:
        Y: shape (N, D-1)
        Z: shape (N, 1)
    """
    rng_key, _ = jax.random.split(rng_key)
    Y = jax.random.uniform(rng_key, shape=(N, D - 1), minval=-1.0, maxval=1.0)
    Y_with_one = jnp.hstack([Y, jnp.ones([Y.shape[0], 1])])
    rng_key, _ = jax.random.split(rng_key)
    beta_true = jax.random.normal(rng_key, shape=(D, 1))
    rng_key, _ = jax.random.split(rng_key)
    Z = Y_with_one @ beta_true + jax.random.normal(rng_key, shape=(N, 1)) * noise
    return Y, Z


def score_fn(X, mu, sigma):
    """
    Computes the score \nabla_y log p(X|mu, sigma), for Stein kernel

    Args:
        X: (N, D)
        mu: (D, )
        sigma: (D, D)
    Returns:
        score: (N, D)
    """
    return -(X - mu[None, :]) @ jnp.linalg.inv(sigma)


def f1(X):
    """
    Args:
        X: (N, D)
    """
    return X.sum(1)


def f1_ground_truth(mu, Sigma):
    return mu.sum()


def f2(X):
    """
    Args:
        X: (N, D)
    """
    D = X.shape[1]
    return 10 * jnp.exp(-0.5 * ((X ** 2).sum(1) / (D ** 1))) + (X ** 2).sum(1)


def f2_ground_truth(mu, Sigma):
    """
    :param mu: (D, )
    :param Sigma: (D, D)
    :return: scalar
    """
    D = mu.shape[0]
    analytical_1 = jnp.exp(-0.5 * mu.T @ jnp.linalg.inv(jnp.eye(D) * (D ** 1) + Sigma) @ mu)
    analytical_2 = jnp.linalg.det(jnp.eye(D) + Sigma / (D ** 1)) ** (-0.5)
    analytical = analytical_1 * analytical_2
    return 10 * analytical + jnp.diag(Sigma).sum() + mu.T @ mu


def f3(X):
    """
    Args:
        X: (N, D)
    """
    return (X ** 2).sum(1)


def f3_ground_truth(mu, Sigma):
    return jnp.diag(Sigma).sum() + mu.T @ mu


def f4(X):
    """
    Only works for D = 2

    Args:
        X: (N, D)
    """
    pred = jnp.array([0.3, 1.0])
    return X @ pred


def f4_ground_truth(mu, Sigma):
    """
    Only for D = 2
    :param mu: (D, )
    :param Sigma: (D, D)
    :return: scalar
    """
    pred = jnp.array([0.3, 1.0])
    return mu.T @ pred


# @jax.jit
def Bayesian_Monte_Carlo_RBF(rng_key, X, f_X, mu_X_theta, var_X_theta, invert_fn=jnp.linalg.inv):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    The kernel_x is RBF, and the hyperparameters are selected by minimizing the negative log-likelihood (NLL).
    Not vectorized over theta.

    Args:
        rng_key: random number generator
        X: shape (N, D)
        f_X: shape (N, )
        var_X_theta: (D, D)
        mu_X_theta: (D, )
        invert_fn: function that inverts a matrix
    Returns:
        I_BQ_mean: float
        I_BQ_std: float
    """
    N, D = X.shape[0], X.shape[1]
    eps = 1e-6

    l_array = jnp.array([0.1, 0.6, 1.0, 3.0]) * D
    nll_array = 0 * l_array
    A_list = []

    for i, l in enumerate(l_array):
        K_no_scale = my_RBF(X, X, l)
        A = f_X.T @ K_no_scale @ f_X / N
        A_list.append(A)
        K = A * K_no_scale
        K_inv = invert_fn(K + eps * jnp.eye(N))
        nll = -(-0.5 * f_X.T @ K_inv @ f_X - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / N
        nll_array = nll_array.at[i].set(nll)

    if D > 2:
        l = l_array[nll_array.argmin()]
        A = A_list[nll_array.argmin()]
    else:
        A = 1.
        l = 1.

    K = A * my_RBF(X, X, l)
    K_inv = invert_fn(K + eps * jnp.eye(N))
    phi = A * kme_RBF_Gaussian(mu_X_theta, var_X_theta, l, X)
    varphi = A * kme_double_RBF_Gaussian(mu_X_theta, var_X_theta, l)

    I_BQ_mean = phi.T @ K_inv @ f_X
    I_BQ_std = jnp.sqrt(jnp.abs(varphi - phi.T @ K_inv @ phi))
    pause = True
    return I_BQ_mean, I_BQ_std


def Bayesian_Monte_Carlo_Matern(rng_key, u, X, f_X, mu_X_theta, var_X_theta):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    The kernel_x is Matern, and the hyperparameters are selected by minimizing the negative log-likelihood (NLL).
    Not vectorized over theta.
    Only works for D = 2.

    Args:
        rng_key: random number generator
        u: shape (N, D), used for reparameterization in Matern kernel. Details in the appendix C.
        X: shape (N, D)
        f_X: shape (N, )
        var_X_theta: (D, D)
        mu_X_theta: (D, )

    Returns:
        I_BQ_mean: float
        I_BQ_std: float
    """
    N, D = X.shape[0], X.shape[1]
    eps = 1e-6

    A = 1.
    l = 1.

    u1 = u[:, 0][:, None]
    u2 = u[:, 1][:, None]

    K = A * my_Matern(u1, u1, l) + A * my_Matern(u2, u2, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = A * kme_Matern_Gaussian(l, u1) + A * kme_Matern_Gaussian(l, u2)
    varphi = phi.mean()

    I_BQ_mean = phi.T @ K_inv @ f_X
    I_BQ_std = jnp.sqrt(jnp.abs(varphi - phi.T @ K_inv @ phi))

    I_BQ_mean = I_BQ_mean.squeeze()
    I_BQ_std = I_BQ_std.squeeze()
    pause = True
    return I_BQ_mean, I_BQ_std


def Bayesian_Monte_Carlo_RBF_vectorized_on_T(rng_key, X, f_X, mu_X_theta, var_X_theta):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    Vectorized over Theta.

    Args:
        rng_key: random number generator
        X: shape (T, N, D)
        f_X: shape (T, N)
        var_X_theta: (T, D, D)
        mu_X_theta: (T, D)
    Returns:
        I_BQ_mean: (T, )
        I_BQ_std: (T, )
    """
    def single_instance(X_single, f_X_single, mu_X_theta_single, var_X_theta_single):
        return Bayesian_Monte_Carlo_RBF(rng_key, X_single, f_X_single, mu_X_theta_single, var_X_theta_single)
    vectorized_function = jax.vmap(single_instance)
    return vectorized_function(X, f_X, mu_X_theta, var_X_theta)


def Bayesian_Monte_Carlo_Matern_vectorized_on_T(rng_key, u, X, f_X, mu_X_theta, var_X_theta):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    Vectorized over Theta.
    Only works for D = 2.

    Args:
        rng_key: random number generator
        X: shape (T, N, D)
        f_X: shape (T, N)
        var_X_theta: (T, D, D)
        mu_X_theta: (T, D)
    Returns:
        I_BQ_mean: (T, )
        I_BQ_std: (T, )
    """
    def single_instance(u_single, X_single, f_X_single, mu_X_theta_single, var_X_theta_single):
        return Bayesian_Monte_Carlo_Matern(rng_key, u_single, X_single, f_X_single, mu_X_theta_single, var_X_theta_single)
    vectorized_function = jax.vmap(single_instance)
    return vectorized_function(u, X, f_X, mu_X_theta, var_X_theta)


def Bayesian_Monte_Carlo_RBF_vectorized_on_T_test(args, rng_key, X, f_X, mu_X_theta_test, var_X_theta_test):
    """
    The BQ baseline, also described as putting a GP prior directly on (x, \theta) -> f(x, \theta)
    Use nystrom to accelerate the computation of matrix inversion.
    Vectorized over Theta_test.

    Args:
        args: arguments
        rng_key: random number generator
        X: shape (N * T, D)
        f_X: shape (N * T, )
        var_X_theta: (T_test, D, D)
        mu_X_theta: (T_test, D)

    Returns:
        BQ_mean: float
        BQ_std: float
    """
    if args.nystrom:
        invert_fn = nystrom_inv
    else:
        invert_fn = jnp.linalg.inv
    def single_instance(mu_single, var_single):
        return Bayesian_Monte_Carlo_RBF(rng_key, X, f_X, mu_single, var_single, invert_fn)

    vectorized_function = jax.vmap(single_instance)
    return vectorized_function(mu_X_theta_test, var_X_theta_test)



# @jax.jit
def GP(rng_key, I_mean, I_std, Theta, Theta_test, eps, kernel_fn):
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
    l_array = jnp.array([0.3, 1.0, 2.0, 3.0]) * D

    A_array = 0 * l_array
    nll_array = jnp.zeros([len(l_array), 1])

    for i, l in enumerate(l_array):
        K_no_scale = kernel_fn(Theta, Theta, l)
        A = I_mean.T @ K_no_scale @ I_mean / T
        A_array = A_array.at[i].set(A)
        K = A * kernel_fn(Theta, Theta, l) + eps * jnp.eye(T) + jnp.diag(I_std ** 2)
        K_inv = jnp.linalg.inv(K)
        nll = -(-0.5 * I_mean.T @ K_inv @ I_mean - 0.5 * jnp.log(jnp.linalg.det(K) + 1e-6)) / T
        nll_array = nll_array.at[i].set(nll)

    l = l_array[jnp.argmin(nll_array)]
    A = A_array[jnp.argmin(nll_array)]

    K_train_train = A * kernel_fn(Theta, Theta, l) + eps * jnp.eye(T) + jnp.diag(I_std ** 2)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = A * kernel_fn(Theta_test, Theta, l)
    K_test_test = A * kernel_fn(Theta_test, Theta_test, l) + eps * jnp.eye(Theta_test.shape[0])

    mu_Theta_test = K_test_train @ K_train_train_inv @ I_mean
    var_Theta_test = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    var_Theta_test = jnp.abs(var_Theta_test)
    std_Theta_test = jnp.sqrt(var_Theta_test)
    pause = True
    return mu_Theta_test, std_Theta_test


def main(args):
    seed = args.seed
    rng_key = jax.random.PRNGKey(seed)
    D = args.dim
    prior_cov_base = 2.0
    noise = 1.0
    T_test = 100
    data_number = 5
    # theta is (N, D-1), X is (N, 1)
    rng_key, _ = jax.random.split(rng_key)
    Y, Z = generate_data(rng_key, D, data_number, noise)

    if args.fn == 'f1':
        f = f1
        f_ground_truth_fn = f1_ground_truth
    elif args.fn == 'f2':
        f = f2
        f_ground_truth_fn = f2_ground_truth
    elif args.fn == 'f3':
        f = f3
        f_ground_truth_fn = f3_ground_truth
    elif args.fn == 'f4':
        f = f4
        f_ground_truth_fn = f4_ground_truth
    else:
        raise ValueError('must be f1 or f2 or f3 or f4!')

    T_array = jnp.array([50, 100])
    # T_array = jnp.concatenate((jnp.array([3, 5]), jnp.arange(10, 150, 10)))
    # T_array = jnp.array([1000])
    N_array = jnp.array([10, 50, 100])
    # N_array = jnp.array([1000])
    # N_array = jnp.concatenate((jnp.array([3, 5]), jnp.arange(10, 150, 10)))

    # This is the test point
    Theta_test = jax.random.uniform(rng_key, shape=(T_test, D), minval=-1.0, maxval=1.0)
    prior_cov_test = jnp.array([[prior_cov_base] * D]) + Theta_test
    ground_truth = jnp.zeros(T_test)

    mu_x_theta_test, var_x_theta_test = sensitivity_utils.posterior_full(Y, Z, prior_cov_test, noise)
    for i in range(T_test):
        ground_truth = ground_truth.at[i].set(f_ground_truth_fn(mu_x_theta_test[i, :], var_x_theta_test[i, :, :]))
    jnp.save(f"{args.save_path}/Theta_test.npy", Theta_test)
    jnp.save(f"{args.save_path}/ground_truth.npy", ground_truth)

    for T in T_array:
        rng_key, _ = jax.random.split(rng_key)
        if args.qmc:
            Theta = sensitivity_utils.qmc_uniform(-1.0, 1.0, D, T)
        else:
            Theta = jax.random.uniform(rng_key, shape=(T, D), minval=-1.0, maxval=1.0)

        for j, N in enumerate(tqdm(N_array)):
            # ======================================== Collecting samples and function evaluations ========================================
            I_BQ_mean_array = jnp.zeros(T)
            I_BQ_std_array = jnp.zeros(T)
            I_MC_mean_array = jnp.zeros(T)
            I_MC_std_array = jnp.zeros(T)

            # This is X, size T * N * D
            X = jnp.zeros([T, N, D]) + 0.0
            # This is f(X), size T * N
            f_X = jnp.zeros([T, N]) + 0.0
            # This is U, size T * N * D, used for CBQ with Matern kernel
            U = jnp.zeros([T, N, D]) + 0.0
            # This is score, size T * N * D, used for CBQ with Stein kernel
            score_all = jnp.zeros([T, N, D]) + 0.0

            prior_cov = jnp.array([[prior_cov_base] * D]) + Theta
            mu_x_theta_all, var_x_theta_all = sensitivity_utils.posterior_full(Y, Z, prior_cov, noise)

            for i in range(T):
                rng_key, _ = jax.random.split(rng_key)
                if args.qmc:
                    X_i, u = sensitivity_utils.qmc_gaussian(mu_x_theta_all[i, :], var_x_theta_all[i, :, :], N)
                    score = score_fn(X_i, mu_x_theta_all[i, :], var_x_theta_all[i, :, :])
                else:
                    u = jax.random.multivariate_normal(rng_key, mean=jnp.zeros_like(mu_x_theta_all[i, :]),
                                                       cov=jnp.diag(jnp.ones_like(mu_x_theta_all[i, :])),
                                                       shape=(N,))
                    L = jnp.linalg.cholesky(var_x_theta_all[i, :, :])
                    X_i = mu_x_theta_all[i, :] + jnp.matmul(L, u.T).T
                    score = score_fn(X_i, mu_x_theta_all[i, :], var_x_theta_all[i, :, :])
                score_all = score_all.at[i, :, :].set(score)
                U = U.at[i, :, :].set(u)
                X = X.at[i, :, :].set(X_i)
                f_X = f_X.at[i, :].set(f(X_i))
            # ======================================== Collecting samples and function evaluations Ends ========================================

            # ======================================== Debug code ========================================
            # for i in range(T):
            #     X_i = X[i, :, :]
            #     f_X_i = f_X[i, :]
            #     u_i = u_all[i, :, :]
            #     score_i = score_all[i, :, :]
            #     mu_X_theta_i = mu_x_theta_all[i, :]
            #     var_X_theta_i = var_x_theta_all[i, :, :]

            #     if args.kernel_x == "RBF":
            #         I_BQ_mean, I_BQ_std = Bayesian_Monte_Carlo_RBF(rng_key, X_i, f_X_i, mu_X_theta_i, var_X_theta_i)
            #     elif args.kernel_x == "Matern":
            #         if D > 2:
            #             raise NotImplementedError("Matern kernel is only implemented for D=2")
            #         I_BQ_mean, I_BQ_std = Bayesian_Monte_Carlo_Matern(rng_key, u_i, X_i, f_X_i, mu_X_theta_i,
            #                                                           var_X_theta_i)
            #     else:
            #         raise NotImplementedError("Kernel not implemented")

            #     I_BQ_mean_array = I_BQ_mean_array.at[i].set(I_BQ_mean)
            #     I_BQ_std_array = I_BQ_std_array.at[i].set(I_BQ_std if not jnp.isnan(I_BQ_std) else 0.01)

            #     I_MC_mean_array = I_MC_mean_array.at[i].set(f_X_i.mean())
            #     I_MC_std_array = I_MC_std_array.at[i].set(f_X_i.std())

            #     true_value = f_ground_truth_fn(mu_X_theta_i, var_X_theta_i)
            #     CBQ_value = I_BQ_mean
            #     print("=============")
            #     print('True value', true_value)
            #     print(f'MC with N={N}', MC_value)
            #     print(f'CBQ with N={N}', CBQ_value)
            #     print(f'CBQ uncertainty {I_BQ_std}')
            #     print(f"=============")
            #     pause = True
            #     ======================================== Debug code ========================================

            # ======================================== CBQ ========================================
            time0 = time.time()
            if args.kernel_x == "rbf":
                I_BQ_mean_array, I_BQ_std_array = Bayesian_Monte_Carlo_RBF_vectorized_on_T(rng_key, X, f_X, mu_x_theta_all, var_x_theta_all)
            elif args.kernel_x == "matern":
                if D > 2:
                    raise NotImplementedError("Matern kernel is only implemented for D=2")
                I_BQ_mean_array, I_BQ_std_array = Bayesian_Monte_Carlo_Matern_vectorized_on_T(rng_key, u, X, f_X, mu_x_theta_all, var_x_theta_all)
            else:
                raise NotImplementedError("Kernel not implemented")
            time_CBQ = time.time() - time0 

            rng_key, _ = jax.random.split(rng_key)
            _, _ = GP(rng_key, I_BQ_mean_array, I_BQ_std_array, Theta, Theta_test, eps=I_BQ_std_array.mean(), kernel_fn=my_Matern)
            
            time0 = time.time()
            if args.kernel_theta == "rbf":
                CBQ_mean, CBQ_std = GP(rng_key, I_BQ_mean_array, I_BQ_std_array, Theta, Theta_test, eps=I_BQ_std_array.mean(), kernel_fn=my_RBF)
            elif args.kernel_theta == "matern":
                CBQ_mean, CBQ_std = GP(rng_key, I_BQ_mean_array, I_BQ_std_array, Theta, Theta_test, eps=I_BQ_std_array.mean(), kernel_fn=my_Matern)
            else:
                raise NotImplementedError(f"Unknown kernel {args.kernel_theta}")
            time_CBQ += time.time() - time0
            # ======================================== CBQ ========================================

            # ======================================== KMS ========================================
            time0 = time.time()
            I_MC_mean_array = f_X.mean(1)
            I_MC_std_array = f_X.std(1)
            if args.baseline_use_variance:
                KMS_mean, KMS_std = baselines.kernel_mean_shrinkage(rng_key, I_MC_mean_array, I_MC_std_array, Theta, Theta_test, eps=0., kernel_fn=my_RBF)
            else:
                KMS_mean, KMS_std = baselines.kernel_mean_shrinkage(rng_key, I_MC_mean_array, None, Theta, Theta_test, eps=0., kernel_fn=my_RBF)
            time_KMS = time.time() - time0
            # ======================================== KMS ========================================

            # ======================================== MOBQ ========================================
            # time0 = time.time()
            # rng_key, _ = jax.random.split(rng_key)
            # BQ_mean, BQ_std = Bayesian_Monte_Carlo_RBF_vectorized_on_T_test(args, rng_key, X.reshape([N * T, D]), f_X.reshape([N * T]), 
            #                                 mu_x_theta_test, var_x_theta_test)
            # time_BQ = time.time() - time0
            # ======================================== MOBQ ========================================

            # ======================================== LSMC ========================================
            time0 = time.time()
            if args.baseline_use_variance:
                LSMC_mean, LSMC_std = baselines.polynomial(Theta, X, f_X, Theta_test, baseline_use_variance=True)
            else:
                LSMC_mean, LSMC_std = baselines.polynomial(Theta, X, f_X, Theta_test, baseline_use_variance=False)
            time_LSMC = time.time() - time0
            # ======================================== LSMC ========================================

            # ======================================== IS ========================================
            time0 = time.time()
            # This is unvectorized version of importance sampling
            # log_px_theta_fn = partial(posterior_log_llk, Y=Y, Z=Z, noise=noise, prior_cov_base=prior_cov_base)
            # IS_mean, IS_std = baselines.importance_sampling(log_px_theta_fn, Theta_test, Theta, X, f_X)

            # This is vectorized version of importance sampling
            log_px_theta_fn_vectorized = partial(sensitivity_utils.posterior_log_llk_vectorized, Y=Y, Z=Z, noise=noise, prior_cov_base=prior_cov_base)
            IS_mean, IS_std = baselines.importance_sampling_sensitivity(log_px_theta_fn_vectorized, Theta_test, Theta, X, f_X)
            time_IS = time.time() - time0
            # ======================================== IS ========================================

            rmse_CBQ, rmse_KMS, rmse_LSMC, rmse_IS = sensitivity_utils.compute_rmse(ground_truth, CBQ_mean, KMS_mean, LSMC_mean, IS_mean)
                                                                                     
            calibration = sensitivity_utils.calibrate(ground_truth, CBQ_mean, jnp.diag(CBQ_std))

            sensitivity_utils.save(args, T, N, rmse_CBQ, rmse_KMS, rmse_LSMC, rmse_IS,
                                   time_CBQ, time_KMS, time_LSMC, time_IS, calibration)

    return


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/sensitivity_conjugate/'
    args.save_path += f"seed_{args.seed}__dim_{args.dim}__function_{args.fn}__Kx_{args.kernel_x}__Ktheta_{args.kernel_theta}"
    args.save_path += f"__qmc_{args.qmc}__usevar_{args.baseline_use_variance}__nystrom_{args.nystrom}_coeff_0.5"
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
