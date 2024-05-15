import jax.numpy as jnp
import numpy as np
import jax
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from kernels import *
import optax
from tqdm import tqdm
from functools import partial
from utils import SIR_utils
from tensorflow_probability.substrates import jax as tfp
from jax.config import config
import baselines
import argparse
import os
import pwd
import shutil

tfd = tfp.distributions
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
    parser.add_argument('--mode', type=str, default='peak_number')
    parser.add_argument('--baseline_use_variance', action='store_true', default=False)
    args = parser.parse_args()
    return args


def Bayesian_Monte_Carlo_vectorized(rng_key, X, f_X, d_log_pX):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    Use stein kernel.
    Not vectorized.

    Args:
        rng_key: random number generator
        x: shape (T, N, D)
        fx: shape (T, N)
        d_log_px: shape (T, N, D)

    Returns:
        I_BQ_mean: (T, )
        I_BQ_std: (T, )
    """
    rng_key, _ = jax.random.split(rng_key)
    vmap_func = jax.vmap(Bayesian_Monte_Carlo, in_axes=(None, 0, 0, 0, None))
    I_BQ_mean, I_BQ_std = vmap_func(rng_key, X, f_X, d_log_pX, stein_Matern)
    return I_BQ_mean, I_BQ_std


# @partial(jax.jit, static_argnums=(4,))
def Bayesian_Monte_Carlo(rng_key, x, fx, d_log_px, kernel_x):
    """
    First stage of CBQ, computes the posterior mean and variance of the integral for a single instance of theta.
    Use stein kernel.
    Not vectorized.

    Args:
        rng_key: random number generator
        x: shape (N, D)
        fx: shape (N, )
        d_log_px: shape (N, D)
        kernel_x: kernel function

    Returns:
        I_BQ_mean: float
        I_BQ_std: float
    """
    x_standardized, x_mean, x_std = SIR_utils.standardize(x)
    x_standardized = x_standardized[:, None]
    fx_scale, fx_standardized = SIR_utils.scale(fx)
    d_log_px = d_log_px * x_std

    eps = 1e-6
    c_init = c = 1.0
    l_init = l = 1.5
    A_init = A = 1.0 / jnp.sqrt(x.shape[0])

    K = A * kernel_x(x_standardized, x_standardized, l, d_log_px, d_log_px) + c + A * jnp.eye(x.shape[0])
    K_inv = jnp.linalg.inv(K)
    I_BQ_mean = c * (K_inv @ fx_standardized).sum()
    I_BQ_std = jnp.sqrt(c - K_inv.sum() * c * c)
    I_BQ_mean = I_BQ_mean * fx_scale
    I_BQ_mean = jnp.where(I_BQ_mean > 2 * fx.mean(0), fx.mean(0), I_BQ_mean)
    pause = True
    return I_BQ_mean, I_BQ_std


# @jax.jit
def prior(X, theta, rate, rng_key):
    """
    Computes p(x | theta) for importance sampling.

    Args:
        X: shape (N, 1)
        theta: scalar 

    Returns:
        likelihood: shape (N, 1)
    """
    scale = 1. / rate
    pdf = 1. / scale * jax.scipy.stats.gamma.pdf(X / scale, a=1 + rate * theta)
    return pdf


@jax.jit
def log_prior(X, theta, rate, rng_key):
    """
    Computes log p(x | theta) for importance sampling.

    Args:
        X: shape (N, 1)
        theta: scalar 

    Returns:
        likelihood: shape (N, 1)
    """   
    scale = 1. / rate
    logpdf = -jnp.log(scale) + jax.scipy.stats.gamma.logpdf(X / scale, a=1 + rate * theta)
    return logpdf


# @jax.jit
def GP(I_mean, I_std, Theta, Theta_test, ground_truth):
    """
    Second stage of CBQ, computes the posterior mean and variance of I(Theta_test).

    Args:
        rng_key: random number generator
        I_mean: (T, )
        I_std: (T, )
        Theta: (T, D)
        Theta_test: (T_test, D)

    Returns:
        mu_Theta_test: (T_test, )
        std_Theta_test: (T_test, )
    """
    eps = 1e-6
    T = I_mean.shape[0]
    Mu_standardized, Mu_mean, Mu_std = SIR_utils.standardize(I_mean)
    Theta_standardized, Theta_mean, Theta_std = SIR_utils.standardize(Theta)
    Theta_test_standardized = (Theta_test - Theta_mean) / Theta_std

    l_array = jnp.array([5.0])
    nll_array = 0.0 * l_array
    A_array = 0.0 * l_array
    sigma = I_std / Mu_std
    for i, l in enumerate(l_array):
        K_no_scale = my_Matern(Theta_standardized, Theta_standardized, l)
        A = Mu_standardized.T @ K_no_scale @ Mu_standardized / T
        A_array = A_array.at[i].set(A)
        K = A * K_no_scale
        K_inv = jnp.linalg.inv(K + jnp.diag(sigma ** 2))
        nll = -(-0.5 * Mu_standardized.T @ K_inv @ Mu_standardized - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / T
        nll_array = nll_array.at[i].set(nll)
    l = l_array[nll_array.argmin()]
    A = A_array[nll_array.argmin()]

    if T > 10:
        sigma = jnp.ones_like(Mu_standardized) * 0.1
    else:
        pass
    K_train_train = A * my_Matern(Theta_standardized, Theta_standardized, l) + jnp.diag(sigma ** 2)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = A * my_Matern(Theta_test_standardized, Theta_standardized, l)
    K_test_test = A * my_Matern(Theta_test_standardized, Theta_test_standardized, l) + (sigma ** 2).mean()
    mu_Theta_test = K_test_train @ K_train_train_inv @ Mu_standardized
    var_Theta_test = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    std_Theta_test = jnp.sqrt(var_Theta_test)

    mu_Theta_test_original = mu_Theta_test * Mu_std + Mu_mean
    std_Theta_test_original = std_Theta_test * Mu_std
    # ======================================== Debug code ========================================
    # plt.figure()
    # plt.plot(Theta_test.squeeze(), ground_truth, color='black')
    # plt.plot(Theta_test.squeeze(), mu_Theta_test_original.squeeze(), color='red')
    # plt.scatter(Theta.squeeze(), I_mean.squeeze(), color='blue')
    # plt.show()
    # pause = True
    # ======================================== Debug code ========================================
    return mu_Theta_test_original, std_Theta_test_original


def peak_infected_number(infections):
    return infections.max(-1)


def SIR(args, rng_key):
    N_array = jnp.array([30, 50, 70])
    T_array = jnp.array([15, 25, 35])
    # T_test = 3 
    # Used for debugging
    T_test = 100 

    population = float(1e5)
    X_real, gamma_real = 0.25, 0.05
    Theta_test = jnp.linspace(0.1, 0.8, T_test)
    rate = 10.0
    scale = 1. / rate
    Time = 150
    dt = 1.0
    SamplesNum = 100

    generate_data_fn = partial(SIR_utils.generate_data, gamma=gamma_real, population=population,
                               Time=Time, dt=dt, rng_key=rng_key)
    generate_data_vmap = jax.vmap(generate_data_fn)

    f = peak_infected_number

    # ======================================== Generate ground truth with large number of samples ========================================
    ground_truth_array = jnp.zeros([T_test])
    for i in tqdm(range(T_test)):
        theta = Theta_test[i]
        a = 1 + rate * theta
        rng_key, _ = jax.random.split(rng_key)
        X_samples = jax.random.gamma(rng_key, a, shape=(SamplesNum,))
        X_samples = X_samples * scale
        temp = generate_data_vmap(X_samples)
        ground_truth_array = ground_truth_array.at[i].set(f(temp).mean())
    # ======================================== Generate ground truth with large number of samples ========================================

    for T in T_array:
        Theta_array = jnp.linspace(0.1, 0.8, T)
        for N in N_array:
            # ======================================== Collecting samples and function evaluations ========================================
            I_BQ_mean_array = jnp.zeros([T])
            I_BQ_std_array = jnp.zeros([T])

            X_array = jnp.zeros([T, N])
            f_X_array = jnp.zeros([T, N])
            d_log_pX_array = jnp.zeros([T, N, 1])

            for j in tqdm(range(T)):
                theta = Theta_array[j]
                a = 1 + rate * theta

                rng_key, _ = jax.random.split(rng_key)
                samples = jax.random.gamma(rng_key, a, shape=(SamplesNum,))
                samples = samples * scale
                rng_key, _ = jax.random.split(rng_key)
                indices = jax.random.permutation(rng_key, jnp.arange(SamplesNum))[:N]
                X = samples[indices]

                log_prior_fn = partial(log_prior, theta=theta, rate=rate, rng_key=rng_key)
                grad_log_prior_fn = jax.grad(log_prior_fn)

                f_X = jnp.zeros([N])
                d_log_pX = jnp.zeros([N, 1])

                for i in range(N):
                    x = X[i]
                    rng_key, _ = jax.random.split(rng_key)
                    D = SIR_utils.generate_data(x, gamma_real, Time, dt, population, rng_key)
                    f_x = f(D)
                    d_log_x = grad_log_prior_fn(x)
                    f_X = f_X.at[i].set(f_x)
                    d_log_pX = d_log_pX.at[i, :].set(d_log_x)

                X_array = X_array.at[j, :].set(X)
                f_X_array = f_X_array.at[j, :].set(f_X)
                d_log_pX_array = d_log_pX_array.at[j, :, :].set(d_log_pX)

                # ======================================== Debug code ========================================
                # _, _ = Bayesian_Monte_Carlo(rng_key, X, f_X, d_log_pX, stein_Matern)
                # tt0 = time.time()
                # I_BQ_mean, I_BQ_std = Bayesian_Monte_Carlo(rng_key, X, f_X, d_log_pX, stein_Matern)
                # tt1 = time.time()

                # I_BQ_mean = I_BQ_mean
                # if I_BQ_mean > 2 * f_X.mean(0):
                #     I_BQ_mean = f_X.mean(0)

                # I_BQ_mean_array = I_BQ_mean_array.at[j].set(I_BQ_mean)
                # I_BQ_std_array = I_BQ_std_array.at[j].set(I_BQ_std)

                # large_samples = generate_data_vmap(samples)
                # f_X_MC_large_sample = f(large_samples).mean()
                # print(f'True value (MC with {SamplesNum} samples)', f_X_MC_large_sample)
                # print(f'MC with {N} number of Y', I_MC)
                # print(f'CBQ with {N} number of Y', I_BQ_mean)
                # print(f"=================")
                # pause = True
                # ======================================== Debug code ========================================

            # ======================================== Collecting samples and function evaluations Ends ========================================

            # ======================================== CBQ ========================================
            I_BQ_mean_array, I_BQ_std_array = Bayesian_Monte_Carlo_vectorized(rng_key, X_array, f_X_array, d_log_pX_array)

            I_BQ_std_array = jnp.nan_to_num(I_BQ_std_array, nan=0.)
            I_BQ_std_array = jnp.ones_like(I_BQ_std_array) * jnp.mean(I_BQ_std_array)

            _, _ = GP(I_BQ_mean_array, I_BQ_std_array, Theta_array[:, None], Theta_test[:, None], ground_truth_array)
            t0 = time.time()
            I_CBQ_mean, I_CBQ_std = GP(I_BQ_mean_array, I_BQ_std_array, Theta_array[:, None], Theta_test[:, None], ground_truth_array)
            I_CBQ_std = jnp.diag(I_CBQ_std)
            time_CBQ = time.time() - t0
            # ======================================== CBQ ========================================

            # ======================================== Importance sampling ========================================
            log_px_theta_fn = partial(log_prior, rate=rate, rng_key=rng_key)
            t0 = time.time()
            IS_mean, IS_std = baselines.importance_sampling_SIR(log_px_theta_fn, Theta_test, Theta_array, X_array, f_X_array)
            time_IS = time.time() - t0
            # ======================================== Importance sampling ========================================

            # ======================================== LSMC ========================================
            t0 = time.time()
            if args.baseline_use_variance:
                LSMC_mean, LSMC_std = baselines.polynomial(Theta_array[:, None], X_array[:, None], f_X_array, Theta_test[:, None], baseline_use_variance=True)
            else:
                LSMC_mean, LSMC_std = baselines.polynomial(Theta_array[:, None], X_array[:, None], f_X_array, Theta_test[:, None], baseline_use_variance=False)
            # ======================================== LSMC ========================================
            time_LSMC = time.time() - t0

            # ======================================== KMS ========================================
            t0 = time.time()
            I_MC_mean_array = f_X_array.mean(1)
            I_MC_std_array = f_X_array.std(1)
            if args.baseline_use_variance:
                KMS_mean, KMS_std = baselines.kernel_mean_shrinkage(rng_key, I_MC_mean_array, I_MC_std_array, Theta_array[:, None], Theta_test[:, None], eps=0., kernel_fn=my_RBF)
            else:
                KMS_mean, KMS_std = baselines.kernel_mean_shrinkage(rng_key, I_MC_mean_array, None, Theta_array[:, None], Theta_test[:, None], eps=0., kernel_fn=my_RBF)
            time_KMS = time.time() - t0
            # ======================================== KMS ========================================

            calibration = SIR_utils.calibrate(ground_truth_array, I_CBQ_mean, I_CBQ_std)

            SIR_utils.save(args, T, N, I_CBQ_mean, KMS_mean, LSMC_mean, IS_mean, ground_truth_array, time_CBQ, time_KMS, time_LSMC, time_IS, calibration)


    return


def main(args):
    # seed = int(time.time())
    seed = args.seed
    rng_key = jax.random.PRNGKey(seed)
    SIR(args, rng_key)
    return


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/SIR/'
    args.save_path += f"seed_{args.seed}__mode_{args.mode}"
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.save_path + '/figures/', exist_ok=True)
    return args


if __name__ == '__main__':
    args = get_config()
    args = create_dir(args)
    print(f'seed is {args.seed}')
    main(args)
    save_path = args.save_path
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    if os.path.exists(f"{save_path}__complete"):
        shutil.rmtree(f"{save_path}__complete")
    os.rename(save_path, f"{save_path}__complete")
