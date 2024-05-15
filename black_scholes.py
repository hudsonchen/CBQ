import numpy as np
import matplotlib.pyplot as plt
from jax.scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy import integrate
import time
import jax
import jax.numpy as jnp
import optax
from functools import partial
from tqdm import tqdm
import baselines
from kernels import *
from utils import black_scholes_utils
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
    parser = argparse.ArgumentParser(description='Conditional Bayesian Quadrature for finance data')

    # Data settings
    parser.add_argument('--kernel_theta', type=str)
    parser.add_argument('--kernel_x', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--baseline_use_variance', action='store_true', default=False)
    parser.add_argument('--nystrom', action='store_true', default=False)
    args = parser.parse_args()
    return args


@jax.jit
def grad_x_log_px_theta(x, theta, x_mean, x_scale, sigma, T_finance, t_finance):
    """
    Computes the score \nabla_x log p(x | theta), for Stein kernel
    p(x | theta) is a log normal distribution with mu=-\sigma^2 / 2 * (T_finance - t_finance) and sigma = \sigma^2 (T_finance - x)

    Args:
        X: (N, D)
        mu: (D, )
        sigma: (D, D)
    Returns:
        score: (N, D)
    """
    x = x * x_scale + x_mean
    part1 = (jnp.log(x) + sigma ** 2 * (T_finance - t_finance) / 2 - jnp.log(theta)) / x / (sigma ** 2 * (T_finance - t_finance))
    return (-1. / x - part1) * x_scale


@partial(jax.jit, static_argnames=['Kx'])
def nllk_func(l, c, A, x, fx, d_log_px, Kx, eps):
    n = x.shape[0]
    K = A * Kx(x, x, l, d_log_px, d_log_px) + c + A * jnp.eye(n)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
    nll = -(-0.5 * fx.T @ K_inv @ fx - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / n
    return nll[0][0]


@partial(jax.jit, static_argnames=['optimizer', 'Kx'])
def step(l, c, A, opt_state, optimizer, x, fx, d_log_px, Kx, eps):
    nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1, 2))(l, c, A, x, fx, d_log_px, Kx, eps)
    updates, opt_state = optimizer.update(grads, opt_state, (l, c, A))
    l, c, A = optax.apply_updates((l, c, A), updates)
    return l, c, A, opt_state, nllk_value


def train(theta, x, x_scale, fx, d_log_px, dx_log_px_fn, rng_key, Kx):
    """
    Use empirical Bayes to train the hyperparameters of the Stein kernel (c, l, A)
    The most time-costly part of CBQ with Stein kernel.

    Args:
        theta: scalar
        x: (N, )
        x_scale: scalar
        fx: (N, )
        d_log_px: (N, )
        dx_log_px_fn: function to compute d_log_px
        rng_key: random key
        Kx: kernel function

    Returns:
        c: scalar
        l: scalar
        A: scalar
    """

    rng_key, _ = jax.random.split(rng_key)
    n = x.shape[0]
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    eps = 1e-6

    c_init = c = 1.0
    l_init = l = 2.0
    A_init = A = 1.0 / jnp.sqrt(n)
    opt_state = optimizer.init((l_init, c_init, A_init))

    # ======================================== Debug code ========================================
    # @jax.jit
    # def nllk_func(l, c, A):
    #     # l = jnp.exp(log_l)
    #     n = x.shape[0]
    #     K = A * Kx(x, x, l, d_log_px, d_log_px) + c + A * jnp.exe(n)
    #     K_inv = jnp.linalg.inv(K + eps * jnp.exe(n))
    #     nll = -(-0.5 * gx.T @ K_inv @ gx - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / n
    #     return nll[0][0]
    #
    # @jax.jit
    # def step(l, c, A, opt_state, rng_kex):
    #     nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1, 2))(l, c, A)
    #     updates, opt_state = optimizer.update(grads, opt_state, (l, c, A))
    #     l, c, A = optax.applx_updates((l, c, A), updates)
    #     return l, c, A, opt_state, nllk_value

    # l_debug_list = []
    # c_debug_list = []
    # A_debug_list = []
    # nll_debug_list = []
    for _ in range(10):
        rng_key, _ = jax.random.split(rng_key)
        l, c, A, opt_state, nllk_value = step(l, c, A, opt_state, optimizer, x, fx, d_log_px, Kx, eps)
        # # Debug code
        # if jnp.isnan(nllk_value):
        #     # l = jnp.exp(log_l)
        #     K = A * Kx(x, x, l, d_log_px, d_log_px) + c + jnp.eye(n)
        #     K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
        #     pause = True
        # l_debug_list.append(l)
        # c_debug_list.append(c)
        # A_debug_list.append(A)
        # nll_debug_list.append(nllk_value)

    # fig = plt.figure(figsize=(15, 6))
    # ax_1, ax_2, ax_3, ax_4 = fig.subplots(1, 4)
    # ax_1.plot(l_debug_list)
    # ax_2.plot(c_debug_list)
    # ax_3.plot(A_debug_list)
    # ax_4.plot(nll_debug_list)
    # plt.show()

    # l = jnp.exp(log_l)
    # A = jnp.exp(log_A)
    # x_debug = jnp.linspace(20, 160, 100)[:, None] / x_scale
    # d_log_px_debug = dx_log_px_fn(x_debug, x)
    # K_train_train = stein_Laplace(x, x, l, d_log_px, d_log_px) + c
    # K_train_train_inv = jnp.linalg.inv(K_train_train + eps * jnp.eye(n))
    # K_test_train = stein_Laplace(x_debug, x, l, d_log_px_debug, d_log_px) + c
    # gx_debug = K_test_train @ K_train_train_inv @ gx
    # plt.figure()
    # plt.scatter(x * x_scale, gx)
    # plt.plot(x_debug * x_scale, gx_debug)
    # plt.show()
    # ======================================== Debug code ========================================
    pause = True
    return l, c, A


def Bayesian_Monte_Carlo_no_stein_inner(theta, Xi, fXi, Kx, lx, eps, sigma, T_finance, t_finance, invert_fn=jnp.linalg.inv):
    Xi = Xi[:, None]
    N = Xi.shape[0]
    K = Kx(Xi, Xi, lx, None, None) + eps * jnp.eye(N)
    K_inv = invert_fn(K)
    a = -sigma ** 2 * (T_finance - t_finance) / 2 + jnp.log(theta)
    b = jnp.sqrt(sigma ** 2 * (T_finance - t_finance))
    phi = kme_log_normal_RBF(Xi, lx, a, b)
    varphi = kme_double_log_normal_RBF(lx, a, b)
    mu = phi.T @ K_inv @ fXi
    std = jnp.sqrt(varphi - phi.T @ K_inv @ phi)
    return mu.squeeze(), std.squeeze()


def Bayesian_Monte_Carlo_no_stein_vectorized(rng_key, Theta, X, fX, Kx):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    Use log_rbf kernel
    Vectorized over Theta.
    
    Args:
        rng_key: random number generator
        X: shape (T, N, D)
        fX: shape (T, N)
        Theta: shape (T, D)
        Kx: kernel function
        
    Returns:
        I_BQ_mean: (T, )
        I_BQ_std: (T, )
    """
        
    sigma = 0.3
    T_finance = 2
    t_finance = 1
    eps = 1e-6
    lx = 0.1

    vmap_func = jax.vmap(Bayesian_Monte_Carlo_no_stein_inner, in_axes=(0, 0, 0, None, None, None, None, None, None))
    I_BQ_mean, I_BQ_std = vmap_func(Theta, X, fX, Kx, lx, eps, sigma, T_finance, t_finance)
    return I_BQ_mean, I_BQ_std


# @partial(jax.jit, static_argnums=(0,))
def Bayesian_Monte_Carlo_no_stein(rng_key, Theta, X, fX, Kx):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    Use log_rbf kernel
    Not vectorized over Theta.
    
    Args:
        rng_key: random number generator
        X: shape (T, N, D)
        fX: shape (T, N)
        Theta: shape (T, D)
        Kx: kernel function
        
    Returns:
        I_BQ_mean: (T, )
        I_BQ_std: (T, )
    """
    
    sigma = 0.3
    T_finance = 2
    t_finance = 1

    T = Theta.shape[0]
    N = X.shape[1]
    eps = 1e-6
    I_BQ_std = jnp.zeros(T)
    I_BQ_mean = jnp.zeros(T)
    lx = 0.1
    
    for i in range(T):
        theta = Theta[i]
        Xi = X[i, :][:, None]
        fXi = fX[i, :][:, None]

        K = Kx(Xi, Xi, lx, None, None) + eps * jnp.eye(N)
        K_inv = jnp.linalg.inv(K)
        a = -sigma ** 2 * (T_finance - t_finance) / 2 + jnp.log(theta)
        b = jnp.sqrt(sigma ** 2 * (T_finance - t_finance))
        phi = kme_log_normal_RBF(Xi, lx, a, b)
        varphi = kme_double_log_normal_RBF(lx, a, b)
        mu = phi.T @ K_inv @ fXi
        std = jnp.sqrt(varphi - phi.T @ K_inv @ phi)

        I_BQ_std = I_BQ_std.at[i].set(std.squeeze())
        I_BQ_mean = I_BQ_mean.at[i].set(mu.squeeze())

        # ======================================== Debug code ========================================
        # print('True value', price(Theta[i], 10000, rng_key)[1].mean())
        # print(f'MC with {N} number of Y', fX.mean())
        # print(f'CBQ with {N} number of Y', mu_standardized.squeeze())
        # print(f"========================================")
        # pause = True
        # ======================================== Debug code ========================================
    return I_BQ_mean, I_BQ_std


# @partial(jax.jit, static_argnums=(0,))
def Bayesian_Monte_Carlo_stein(rng_key, Theta, X, fX, Kx):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    Use stein kernel

    Args:
        rng_key: random number generator
        X: shape (T, N, D)
        fX: shape (T, N)
        Theta: shape (T, D)
        Kx: kernel function

    Returns:
        I_BQ_mean: (T, )
        I_BQ_std: (T, )
    """

    T = Theta.shape[0]
    N = X.shape[1]
    eps = 1e-6
    I_BQ_std = jnp.zeros(T)
    I_BQ_mean = jnp.zeros(T)

    for i in range(T):
        theta = Theta[i]
        Xi = X[i, :][:, None]
        Xi_standardized, Xi_mean, Xi_scale = black_scholes_utils.standardize(Xi)
        fXi = fX[i, :][:, None]

        grad_x_log_px_theta_fn = partial(grad_x_log_px_theta, sigma=0.3, T_finance=2, t_finance=1, x_mean=Xi_mean, x_scale=Xi_scale)
        dx_log_px_theta = grad_x_log_px_theta_fn(Xi_standardized, theta)
        if i == 0:
            lx, c, A = train(theta, Xi_standardized, Xi_scale, fXi,
                                dx_log_px_theta, grad_x_log_px_theta_fn, rng_key, Kx)

        K = A * Kx(Xi_standardized, Xi_standardized, lx, dx_log_px_theta, dx_log_px_theta) + c + A * jnp.eye(N)
        K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
        mu = c * (K_inv @ fXi).sum()
        std = jnp.sqrt(c - K_inv.sum() * c ** 2)

        I_BQ_std = I_BQ_std.at[i].set(std.squeeze())
        I_BQ_mean = I_BQ_mean.at[i].set(mu.squeeze())

        # ======================================== Debug code ========================================
        # # Large sample mu
        # print('True value', price(theta[i], 10000, rng_key)[1].mean())
        # print(f'MC with {N} number of Y', fXi.mean())
        # print(f'CBQ with {N} number of Y', mu)
        # print(f"=================")
        # pause = True
        # ======================================== Debug code ========================================
    return I_BQ_mean, I_BQ_std
    

def Bayesian_Monte_Carlo_no_stein_vectorized_on_T_test(args, rng_key, Theta_test, X, fX, Kx):
    """
    The BQ baseline, also described as putting a GP prior directly on (x, \theta) -> f(x, \theta)
    Vectorized over Theta_test.

    Args:
        rng_key: random number generator
        X: shape (N * T, D)
        fX: shape (N * T, )
        Theta_test: shape (T_test, D)
        Kx: kernel function

    Returns:
        BQ_mean: float
        BQ_std: float
    """
    sigma = 0.3
    T_finance = 2
    t_finance = 1
    eps = 1e-6
    lx = 0.1
    if args.nystrom:
        invert_fn = nystrom_inv
    else:
        invert_fn = jnp.linalg.inv
    vmap_func = jax.vmap(Bayesian_Monte_Carlo_no_stein_inner, in_axes=(0, None, None, None, None, None, None, None, None, None))
    BQ_mean, BQ_std = vmap_func(Theta_test, X, fX, Kx, lx, eps, sigma, T_finance, t_finance, invert_fn)
    return BQ_mean, BQ_std


# @partial(jax.jit, static_argnums=(0,))
def GP(I_mean, I_std, Theta, Theta_test, Ktheta):
    """
    Second stage of CBQ, computes the posterior mean and variance of I(Theta_test).

    Args:
        rng_key: random number generator
        I_mean: (T, )
        I_std: (T, )
        Theta: (T, D)
        Theta_test: (T_test, D)
        Ktheta: Matern or RBF

    Returns:
        mu_Theta_test: (T_test, )
        std_Theta_test: (T_test, )
    """
    Theta_standardized, theta_mean, theta_std = black_scholes_utils.standardize(Theta)
    Theta_test_standardized = (Theta_test - theta_mean) / theta_std
    ltheta = 1.5

    noise = I_std.mean()
    K_train_train = Ktheta(Theta_standardized, Theta_standardized, ltheta) + jnp.diag(I_std ** 2)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = Ktheta(Theta_test_standardized, Theta_standardized, ltheta)
    K_test_test = Ktheta(Theta_test_standardized, Theta_test_standardized, ltheta) + noise

    mu_Theta_test = K_test_train @ K_train_train_inv @ I_mean
    var_Theta_test = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    var_Theta_test = jnp.abs(var_Theta_test)
    std_Theta_test = jnp.sqrt(var_Theta_test)
    pause = True
    return mu_Theta_test, std_Theta_test



@partial(jax.jit, static_argnums=(1,))
def price(St, N, rng_key):
    """
    Computes the price ST at time T_finance.
    ST is sampled from the conditional distribution p(ST|St).
    Computes the loss \psi(ST) - \psi((1+s)ST) caused by the shock. 
    Their shape is T * N
    
    Args:
        St: (T, 1) the price at time t_finance
        N: number of samples
        
    Returns:
        ST: (T, N, 1)
        f(ST): (T, N)
    """

    K1 = 50
    K2 = 150
    s = -0.2
    sigma = 0.3
    T_finance = 2
    t_finance = 1

    output_shape = (St.shape[0], N)
    rng_key, _ = jax.random.split(rng_key)
    epsilon = jax.random.normal(rng_key, shape=output_shape)
    ST = St * jnp.exp(sigma * jnp.sqrt((T_finance - t_finance)) * epsilon - 0.5 * (sigma ** 2) * (T_finance - t_finance))
    psi_ST_1 = jnp.maximum(ST - K1, 0) + jnp.maximum(ST - K2, 0) - 2 * jnp.maximum(ST - (K1 + K2) / 2, 0)
    psi_ST_2 = jnp.maximum((1 + s) * ST - K1, 0) + jnp.maximum((1 + s) * ST - K2, 0) - 2 * jnp.maximum(
        (1 + s) * ST - (K1 + K2) / 2, 0)
    return ST, psi_ST_1 - psi_ST_2


def save_true_value(St, args):
    """
    Computes the ground truth value.
    
    Args:
        St: (T, 1) the price at time t_finance
        args: arguments
    """
    seed = args.seed
    rng_key = jax.random.PRNGKey(seed)
    rng_key, _ = jax.random.split(rng_key)

    # K1 = 50
    # K2 = 150
    # s = -0.2
    # t = 1
    # T = 2
    # sigma = 0.3
    # S0 = 50

    _, loss = price(St, 100000, rng_key)
    value = loss.mean(1)
    jnp.save(f"{args.save_path}/finance_theta.npy", St)
    jnp.save(f"{args.save_path}/finance_EfX_theta.npy", value)
    # plt.figure()
    # plt.plot(St, value)
    # plt.xlabel(r"$theta$")
    # plt.ylabel(r"$\mathbb{E}[f(X) \mid theta]$")
    # plt.title("True value for finance experiment")
    # plt.savefig(f"{args.save_path}/true_distribution.pdf")
    # # plt.show()
    # plt.close()
    return


def option_pricing(args):
    seed = args.seed
    # seed = 0
    rng_key = jax.random.PRNGKey(seed)
    rng_key, _ = jax.random.split(rng_key)

    # T_array = jnp.array([10, 20, 30])
    T_array = jnp.array([20])
    N_array = jnp.array([20, 50, 80])

    test_num = 200
    St_test = jnp.linspace(20., 120., test_num)[:, None]
    save_true_value(St_test, args)

    for T in T_array:
        for N in tqdm(N_array):
            # ======================================== Collecting samples and function evaluations ========================================
            rng_key, _ = jax.random.split(rng_key)
            # epsilon = jax.random.normal(rng_key, shape=(T, 1))
            # St = S0 * jnp.exp(sigma * jnp.sqrt(t) * epsilon - 0.5 * (sigma ** 2) * t)
            St = jnp.linspace(20, 120, T)[:, None]
            ST, loss = price(St, N.item(), rng_key)\
            # ======================================== Collecting samples and function evaluations ========================================

            # ======================================== KMS ========================================
            I_MC_mean = loss.mean(1)
            M = 100
            rng_key, _ = jax.random.split(rng_key)
            samples = [loss[:, jax.random.choice(rng_key, N, (N//2, ), replace=False)].mean(1)[:, None] for _ in range(M)]
            stacked_samples = jnp.hstack(samples)
            I_MC_std = jnp.std(stacked_samples, axis=1)

            time0 = time.time()
            if args.baseline_use_variance:
                KMS_mean, KMS_std = baselines.kernel_mean_shrinkage(rng_key, I_MC_mean, I_MC_std, St, St_test, eps=0., kernel_fn=my_RBF)
            else:
                KMS_mean, KMS_std = baselines.kernel_mean_shrinkage(rng_key, I_MC_mean, None, St, St_test, eps=0., kernel_fn=my_RBF)
            time_KMS = time.time() - time0
            # ======================================== KMS ========================================

            # ======================================== IS ========================================
            t0 = time.time()
            IS_mean, IS_std = baselines.importance_sampling_finance(black_scholes_utils.px_theta_fn, St_test, St, ST, loss)
            time_IS = time.time() - t0
            # ======================================== IS ========================================

            # ======================================== LSMC ========================================
            t0 = time.time()
            if args.baseline_use_variance:
                LSMC_mean, LSMC_std = baselines.polynomial(St, ST, loss, St_test, baseline_use_variance=True)
            else:
                LSMC_mean, LSMC_std = baselines.polynomial(St, ST, loss, St_test, baseline_use_variance=False)
            time_LSMC = time.time() - t0
            # ======================================== LSMC ========================================

            # ======================================== CBQ ========================================
            t0 = time.time()
            if 'stein' not in args.kernel_x:
                # I_BQ_mean, I_BQ_std = Bayesian_Monte_Carlo_no_stein(rng_key, St, ST, loss, log_normal_RBF)
                I_BQ_mean, I_BQ_std = Bayesian_Monte_Carlo_no_stein_vectorized(rng_key, St, ST, loss, log_normal_RBF)
            elif 'stein' in args.kernel_x:
                I_BQ_mean, I_BQ_std  = Bayesian_Monte_Carlo_stein(rng_key, St, ST, loss, stein_Matern)
            else:
                raise NotImplementedError(args.kernel_x)
            t1 = time.time()

            I_BQ_std = np.nan_to_num(I_BQ_std, nan=0.3)
            _, _ = GP(I_BQ_mean, I_BQ_std, St, St_test, my_RBF)
            t2 = time.time()
            CBQ_mean, CBQ_std = GP(I_BQ_mean, I_BQ_std, St, St_test, my_RBF)
            t3 = time.time()
            time_CBQ = t3 - t2 + t1 - t0
            # ======================================== CBQ ========================================

            ground_truth = jnp.load(f"{args.save_path}/finance_EfX_theta.npy")
            calibration = black_scholes_utils.calibrate(ground_truth, CBQ_mean, jnp.diag(CBQ_std))

            black_scholes_utils.save(T, N, CBQ_mean, CBQ_std, KMS_mean, IS_mean, LSMC_mean, 
                               time_CBQ, time_IS, time_KMS, time_LSMC, calibration, args.save_path)

    return


def main(args):
    seed = args.seed
    
    # ======================================== Debug code ========================================
    # rng_key = jax.random.PRNGKey(seed)
    # visualize_brownian = False
    # debug_BSM = False
    # if visualize_brownian:
    #     n = 100.
    #     T = 1.
    #     dt = T / n
    #     plt.figure()
    #     for i in range(10):
    #         St = black_scholes_utils.Geometric_Brownian(n, dt, rng_key)
    #         plt.plot(St)
    #     # plt.show()
    # elif debug_BSM:
    #     black_scholes_utils.BSM_butterfly_analytic()
    # else:
    #     pass
    # ======================================== Debug code ========================================
    option_pricing(args)
    return


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    if 'stein' in args.kernel_x:
        args.save_path += f'results/finance_stein/'
    else:
        args.save_path += f'results/finance/'
    args.save_path += f"usevar_{args.baseline_use_variance}__nystrom_{args.nystrom}__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(f"{args.save_path}/figures/", exist_ok=True)
    return args


if __name__ == '__main__':
    args = get_config()
    args = create_dir(args)
    print(f'Device is {jax.devices()}')
    print(args.seed)
    main(args)
    save_path = args.save_path
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    if os.path.exists(f"{save_path}__complete"):
        shutil.rmtree(f"{save_path}__complete")
    os.rename(save_path, f"{save_path}__complete")
    print("\n------------------- DONE -------------------\n")

