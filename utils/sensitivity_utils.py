import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm
from sobol_seq import i4_sobol_generate


# @jax.jit
def posterior_full(Y, Z, prior_cov, noise):
    """
    Computes posterior mean and covariance for Bayesian linear regression

    Args:
        Y: shape (N, D-1)
        Z: shape (N, 1)
        prior_cov: shape (N3, D)
        noise: float
    
    Returns:
        post_mean: shape (N, D)
        post_cov: shape (N, D, D)
    """
    Y_with_one = jnp.hstack([Y, jnp.ones([Y.shape[0], 1])])
    D = prior_cov.shape[-1]
    prior_cov_inv = 1. / prior_cov
    # (N3, D, D)
    prior_cov_inv = jnp.einsum('ij,jk->ijk', prior_cov_inv, jnp.eye(D))
    beta_inv = noise ** 2
    beta = 1. / beta_inv
    post_cov = jnp.linalg.inv(prior_cov_inv + beta * Y_with_one.T @ Y_with_one)
    post_mean = beta * post_cov @ Y_with_one.T @ Z
    return post_mean.squeeze(), post_cov


# @jax.jit
def normal_logpdf(x, mu, cov):
    """
    Computes normal logpdf, a less vectorized version for importance sampling

    Args:
        x: shape (N2, D)
        mu: shape (N3, D)
        cov: shape (N3, D, D)
    
    Returns:
        log likelihood: shape (N2, N3)
    """
    N2, D = x.shape
    N3 = mu.shape[0]

    x_expanded = jnp.expand_dims(x, 1)  # Shape (N2, 1, D)
    mean_expanded = jnp.expand_dims(mu, 0)  # Shape (1, N3, D)

    diff = x_expanded - mean_expanded  # Shape (N2, N3, D)
    precision_matrix = jnp.linalg.inv(cov)  # Shape (N3, D, D)
    exponent = -0.5 * jnp.einsum('nij, njk, nik->ni', diff, precision_matrix, diff)  # Shape (N2, N3)

    normalization = -0.5 * (D * jnp.log(2 * jnp.pi) + jnp.log(jnp.linalg.det(cov)))  # Shape (N3,)
    normalization = jnp.expand_dims(normalization, 0)  # Shape (1, N3)

    return normalization + exponent  # Shape (N2, N3)


# @jax.jit
def normal_logpdf_vectorized(x, mu, cov):
    """
    Computes normal logpdf, a less vectorized version for importance sampling

    Args:
        x: shape (N1, N2, D)
        mu: shape (N3, D)
        cov: shape (N3, D, D)
    
    Returns:
        log likelihood: shape (N1, N2, N3)
    """

    D = x.shape[-1]
    x_expanded = jnp.expand_dims(x, 2)
    mean_expanded = jnp.expand_dims(mu, (0, 1))
    # covariance_expanded = jnp.expand_dims(covariance, 0)

    diff = x_expanded - mean_expanded
    precision_matrix = jnp.linalg.inv(cov)
    exponent = -0.5 * jnp.einsum('nijk, jkl, nijl->nij', diff, precision_matrix, diff)
    normalization = -0.5 * (D * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(jnp.linalg.det(cov)))
    return normalization + exponent


# @jax.jit
def posterior_log_llk_vectorized(X, prior_cov_base, Y, Z, Theta, noise):
    """
    :param X: (N1, N2, D)
    :param prior_cov_base: scalar
    :param Y: data
    :param Z: data
    :param Theta: (N3, D)
    :param noise: scalar
    :return:
    """
    D = X.shape[-1]
    # prior_cov is (N3, D)
    prior_cov = jnp.ones([1, D]) * prior_cov_base + Theta
    # post_mean is (N3, D), post_cov is (N3, D, D)
    post_mean, post_cov = posterior_full(Y, Z, prior_cov, noise)
    return normal_logpdf_vectorized(X, post_mean, post_cov)



# @jax.jit
def posterior_log_llk(X, prior_cov_base, Y, Z, theta, noise):
    """
    :param X: (N2, D)
    :param prior_cov_base: scalar
    :param Y: data
    :param Z: data
    :param theta: (D, )
    :param noise: data noise
    :return:
    """
    D = X.shape[-1]
    # Turn theta into shape (N3, D)
    theta = theta[None, :]
    # prior_cov is (N3, D)
    prior_cov = jnp.ones([1, D]) * prior_cov_base + theta
    # post_mean is (N3, D), post_cov is (N3, D, D)
    post_mean, post_cov = posterior_full(Y, Z, prior_cov, noise)
    return normal_logpdf(X, post_mean, post_cov).squeeze()


def compute_rmse(ground_truth, CBQ_mean, KMS_mean, LSMC_mean, IS_mean):
    rmse_CBQ = jnp.sqrt(jnp.mean((CBQ_mean - ground_truth) ** 2))
    rmse_KMS = jnp.sqrt(jnp.mean((KMS_mean - ground_truth) ** 2))
    rmse_LSMC = jnp.sqrt(jnp.mean((LSMC_mean - ground_truth) ** 2))
    rmse_IS = jnp.sqrt(jnp.mean((IS_mean - ground_truth) ** 2))
    return rmse_CBQ, rmse_KMS, rmse_LSMC, rmse_IS


def save(args, T, N, rmse_CBQ, rmse_KMS, rmse_LSMC, rmse_IS,
         time_CBQ, time_KMS, time_LSMC, time_IS, calibration):
    rmse_dict = {}
    rmse_dict["CBQ"] = rmse_CBQ
    rmse_dict["KMS"] = rmse_KMS
    rmse_dict["LSMC"] = rmse_LSMC
    rmse_dict["IS"] = rmse_IS
    with open(f"{args.save_path}/rmse_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(rmse_dict, f)

    time_dict = {}
    time_dict["CBQ"] = time_CBQ
    time_dict["KMS"] = time_KMS
    time_dict["LSMC"] = time_LSMC
    time_dict["IS"] = time_IS
    with open(f"{args.save_path}/time_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(time_dict, f)

    jnp.save(f"{args.save_path}/calibration_T_{T}_N_{N}", calibration)

    methods = ["CBQ", "KMS", "LSMC", "IS"]
    rmse_values = [rmse_CBQ, rmse_KMS, rmse_LSMC, rmse_IS]
    time_values = [time_CBQ, time_KMS, time_LSMC, time_IS]

    print("\n\n========================================")
    print(f"T = {T} and N = {N}")
    print("========================================")
    print("Methods:    " + " ".join([f"{method:<10}" for method in methods]))
    print("RMSE:       " + " ".join([f"{value:<10.6f}" for value in rmse_values]))
    print("Time (s):   " + " ".join([f"{value:<10.6f}" for value in time_values]))
    print("========================================\n\n")
    return


def save_large(args, T, N, rmse_KMS, rmse_LSMC, rmse_IS, time_KMS, time_LSMC, time_IS):
    rmse_dict = {}
    rmse_dict["CBQ"] = None
    rmse_dict["KMS"] = rmse_KMS
    rmse_dict["LSMC"] = rmse_LSMC
    rmse_dict["IS"] = rmse_IS
    with open(f"{args.save_path}/rmse_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(rmse_dict, f)

    time_dict = {}
    time_dict["CBQ"] = None
    time_dict["KMS"] = time_KMS
    time_dict["LSMC"] = time_LSMC
    time_dict["IS"] = time_IS
    with open(f"{args.save_path}/time_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(time_dict, f)
    return


def scale(Z):
    s = Z.mean()
    standardized = Z / s
    return standardized, s


def standardize(Z):
    mean = Z.mean(0)
    std = Z.std(0)
    standardized = (Z - mean) / std
    return standardized, mean, std


def qmc_gaussian(mu, sigma, nsamples):
    """
    :param mu: (D, )
    :param sigma: (D, D)
    :param nsamples:
    :return: samples: (nsamples, D)
    """
    D = mu.shape[0]
    u = i4_sobol_generate(D, nsamples)
    L = jnp.linalg.cholesky(sigma)
    samples = mu[:, None] + (norm.ppf(u) @ L).T
    samples = samples.T
    return samples, u


def qmc_uniform(min_val, max_val, D, nsamples):
    """
    :param min_val:
    :param max_val:
    :param nsamples:
    :return:
    """
    u = i4_sobol_generate(D, nsamples)
    samples = min_val + u * (max_val - min_val)
    return samples


def calibrate(ground_truth, CBQ_mean, CBQ_std):
    """
    Calibration plot for CBQ.

    Args:
        ground_truth: (T_test, )
        CBQ_mean: (T_test, )
        CBQ_std: (T_test, )
        
    Returns:
        prediction_interval: (21, )
    """
    confidence_level = jnp.arange(0.0, 1.01, 0.05)
    prediction_interval = jnp.zeros(len(confidence_level))
    for i, c in enumerate(confidence_level):
        z_score = norm.ppf(1 - (1 - c) / 2)  # Two-tailed z-score for the given confidence level
        prob = jnp.less(jnp.abs(ground_truth - CBQ_mean), z_score * CBQ_std)
        prediction_interval = prediction_interval.at[i].set(prob.mean())

    # plt.figure()
    # plt.plot(confidence_level, prediction_interval, label="Model calibration", marker="o")
    # plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal calibration", color="black")
    # plt.xlabel("Confidence")
    # plt.ylabel("Coverage")
    # plt.title("Calibration plot")
    # plt.legend()
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.close()
    # plt.show()
    return prediction_interval
