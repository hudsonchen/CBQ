import jax.numpy as jnp
from functools import partial
import jax
from utils import black_scholes_utils


def polynomial(Theta, X, f_X, Theta_test, baseline_use_variance, poly=3):
    """
    Least Squares Monte Carlo (LSMC)

    Args:
        theta: shape (T, D)
        X: shape (T, N, D)
        f_X: shape (T, N)
        theta_test: shape (T_test, D)
        baseline_use_variance: boolean, whether use variance
        poly: order of polynomial, defaults to 3

    Returns:
        I_LSMC_mean: shape (T_test, )
        I_LSMC_std: shape (T_test, )
    """
    powers = jnp.arange(0, poly + 1)
    theta_poly = Theta[:, :, None] ** powers
    theta_poly = theta_poly.reshape([Theta.shape[0], -1])

    if not baseline_use_variance:
        eps = 1e-6
        D_all = (1 + poly) * Theta.shape[1]
        coeff = jnp.linalg.inv(theta_poly.T @ theta_poly + eps * jnp.eye(D_all)) @ theta_poly.T @ f_X.mean(1)
    else:
        eps = 1e-6
        MC_std = f_X.std(1)
        D_all = (1 + poly) * Theta.shape[1]
        coeff = jnp.linalg.inv(theta_poly.T @ jnp.diag(MC_std) @ theta_poly + eps * jnp.eye(D_all)) @ (theta_poly.T @ jnp.diag(MC_std) @ f_X.mean(1))
    
    theta_test_poly = Theta_test[:, :, None] ** powers
    theta_test_poly = theta_test_poly.reshape([Theta_test.shape[0], -1])
    I_LSMC_mean = theta_test_poly @ coeff
    return I_LSMC_mean, 0 * I_LSMC_mean


def importance_sampling(log_px_theta_fn, Theta_test, Theta, X, f_X):
    """
    Importance Sampling, this function is for general use, not being vectorized.

    Args:
        log_px_theta_fn (function): log p(x|theta)
        Theta: shape (T, D)
        X: shape (T, N, D)
        f_X: shape (T, N)
        Theta_test: shape (T_test, D)

    Returns:
        IS_mean: shape (T_test, )
        IS_std: shape (T_test, )
    """
    T, N, D = X.shape
    IS_list = []
    for j in range(len(Theta_test)):
        theta_test = Theta_test[j]
        dummy_list = []
        for i in range(T):
            theta = Theta[i]
            Xi = X[i, :]
            f_X_i = f_X[i, :]
            log_py_theta_i = log_px_theta_fn(X=Xi, theta=theta)
            log_py_theta_test = log_px_theta_fn(X=Xi, theta=theta_test)
            weight = jnp.exp(log_py_theta_test - log_py_theta_i)
            dummy_list.append((weight * f_X_i).sum())
        IS_list.append(jnp.array(dummy_list).mean())
    IS_mean = jnp.array(IS_list)
    return IS_mean, 0 * IS_mean



# @jax.jit
def kernel_mean_shrinkage(rng_key, I_mean, I_std, Theta, Theta_test, eps, kernel_fn):
    """
    Kernel Mean Shrinkage (KMS)
    The hyperparameters are selected by minimizing the negative log-likelihood (NLL) on the training set.

    Args:
        I_mean: shape (T, ), MC mean from stage one
        I_std: shape (T, ), MC std from stage one
        Theta: shape (T, D)
        Theta_test: shape (T_test, D)

    Returns:
        mu_Theta_test: shape (T_test, )
        std_Theta_test: shape (T_test, )
    """
    T, D = Theta.shape[0], Theta.shape[1]
    l_array = jnp.array([0.3, 1.0, 2.0, 3.0]) * D

    if I_std is None:
        sigma_array = jnp.array([1.0, 0.1, 0.01, 0.001])
        A_array = jnp.array([10.0, 100.0, 300.0, 1000.0])
        nll_array = jnp.zeros([len(l_array), len(A_array), len(sigma_array)])
    else:
        sigma_array = jnp.array([0.0])
        A_array = 0 * l_array
        nll_array = jnp.zeros([len(l_array), 1])

    if I_std is None:
        for i, l in enumerate(l_array):
            for j, A in enumerate(A_array):
                for k, sigma in enumerate(sigma_array):
                    K = A * kernel_fn(Theta, Theta, l) + jnp.eye(T) * sigma
                    K_inv = jnp.linalg.inv(K)
                    nll = -(-0.5 * I_mean.T @ K_inv @ I_mean - 0.5 * jnp.log(
                        jnp.linalg.det(K) + 1e-6)) / T
                    nll_array = nll_array.at[i, j].set(nll)
        min_index_flat = jnp.argmin(nll_array)
        i1, i2, i3 = jnp.unravel_index(min_index_flat, nll_array.shape)
        l = l_array[i1]
        A = A_array[i2]
        sigma = sigma_array[i3]
    else:
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

    if I_std is None:
        K_train_train = A * kernel_fn(Theta, Theta, l) + jnp.eye(T) * sigma
        K_train_train_inv = jnp.linalg.inv(K_train_train)
        K_test_train = A * kernel_fn(Theta_test, Theta, l)
        K_test_test = A * kernel_fn(Theta_test, Theta_test, l) + jnp.eye(Theta_test.shape[0]) * sigma
    else:
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


# @partial(jax.jit, static_argnums=(0,))
def importance_sampling_sensitivity(log_pX_theta_fn, Theta_test, Theta, X, f_X):
    """
    Importance Sampling, this function is for bayes sensitivity analysis, fully vectorized.

    Args:
        log_px_theta_fn (function): log p(x|theta)
        Theta: shape (T, D)
        X: shape (T, N, D)
        f_X: shape (T, N)
        Theta_test: shape (T_test, D)

    Returns:
        IS_mean: shape (T_test, )
        IS_std: shape (T_test, )
    """
    # log_pX_theta_test is (T, N, T_Test)
    log_pX_theta_test = log_pX_theta_fn(X=X, Theta=Theta_test)
    # log_pX_theta_i is (T, N, T)
    log_pX_theta_i = log_pX_theta_fn(X=X, Theta=Theta)

    # log_pX_theta_test is (T, N, T_Test)
    log_pX_theta_test = log_pX_theta_test.transpose(2, 0, 1)
    # log_pX_theta_i is (T, T, N)
    log_pX_theta_i = log_pX_theta_i.transpose(2, 0, 1)

    # weight is (T_Test, T, N)
    weight = jnp.exp(log_pX_theta_test - jnp.diagonal(log_pX_theta_i, axis1=0, axis2=1).transpose(1, 0))
    # mu is (T_Test, T)
    mu = (weight * f_X).mean(2)
    IS_mean = mu.mean(1)
    return IS_mean, 0 * IS_mean


def importance_sampling_single_finance(tree, pX_theta_fn, theta_test):
    theta, Xi, f_Xi = tree
    Xi_standardized, Xi_mean, Xi_scale = black_scholes_utils.standardize(Xi)
    pX_theta_standardized_fn = partial(pX_theta_fn, sigma=0.3, T_finance=2, t_finance=1, x_scale=Xi_scale, x_mean=Xi_mean)
    pX_theta_test = pX_theta_standardized_fn(Xi_standardized, theta_test)
    pX_theta_i = pX_theta_standardized_fn(Xi_standardized, theta)
    weight = pX_theta_test / pX_theta_i
    return (weight * f_Xi).mean() / weight.mean()


def importance_sampling_finance_(pX_theta_fn, theta, X, f_X, theta_test):
    theta_test = theta_test[:, None]
    importance_sampling_single_fn = partial(importance_sampling_single_finance, pX_theta_fn=pX_theta_fn, theta_test=theta_test)
    importance_sampling_single_vmap = jax.vmap(importance_sampling_single_fn, in_axes=((0, 0, 0),))
    tree = (theta, X, f_X)
    dummy = importance_sampling_single_vmap(tree)
    return dummy.mean()


def importance_sampling_finance(pX_theta_fn, Theta_test, Theta, X, f_X):
    """
    Importance Sampling, this function is for Black Scholes model, fully vectorized.

    Args:
        pX_theta_fn (function): p(x|theta)
        Theta (jnp.array): shape (T, D)
        X (jnp.array): shape (T, N, D)
        f_X (jnp.array): shape (T, N)
        Theta_test (jnp.array): shape (T_test, D)

    Returns:
        IS_mean: shape (T_test, )
        IS_std: shape (T_test, )
    """
    importance_sampling_fn = partial(importance_sampling_finance_, pX_theta_fn, Theta, X, f_X)
    importance_sampling_vmap = jax.vmap(importance_sampling_fn)
    IS_mean = importance_sampling_vmap(Theta_test)
    return IS_mean, 0 * IS_mean


def importance_sampling_single_SIR(tree, log_py_theta_fn, theta_test):
    theta_i, Xi, f_Xi = tree
    log_py_theta_test = log_py_theta_fn(X=Xi, theta=theta_test)
    log_py_theta_i = log_py_theta_fn(X=Xi, theta=theta_i)
    weight = jnp.exp(log_py_theta_test - log_py_theta_i)
    mu = (weight * f_Xi).mean() / weight.mean()
    return mu


def importance_sampling_SIR_(log_py_theta_fn, X, Theta, f_X, theta_test):
    importance_sampling_single_fn = partial(importance_sampling_single_SIR, log_py_theta_fn=log_py_theta_fn, theta_test=theta_test)
    importance_sampling_single_vmap = jax.vmap(importance_sampling_single_fn, in_axes=((0, 0, 0),))
    tree = (Theta, X, f_X)
    dummy = importance_sampling_single_vmap(tree)
    return dummy.mean()


def importance_sampling_SIR(log_py_theta_fn, Theta_test, Theta, X, f_X):
    """
    Importance Sampling, this function is for SIR model, fully vectorized.

    Args:
        log_py_theta_fn (function): log p(x|theta)
        Theta (jnp.array): shape (T, D)
        X (jnp.array): shape (T, N, D)
        f_X (jnp.array): shape (T, N)
        Theta_test (jnp.array): shape (T_test, D)

    Returns:
        IS_mean: shape (T_test, )
        IS_std: shape (T_test, )
    """
    importance_sampling_fn = partial(importance_sampling_SIR_, log_py_theta_fn, X, Theta, f_X)
    importance_sampling_vmap = jax.vmap(importance_sampling_fn)
    IS_mean = importance_sampling_vmap(Theta_test)
    return IS_mean, 0 * IS_mean

