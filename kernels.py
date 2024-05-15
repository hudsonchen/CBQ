import jax.numpy as jnp
import jax
from tensorflow_probability.substrates import jax as tfp
from functools import partial
from jax.scipy.stats import norm
import time


def stein_Matern(x, y, l, d_log_px, d_log_py):
    """
    Stein Matern kernel.

    Args:
        x: (N, D)
        y: (M, D)
        l: scalar
        d_log_px: (N, D)
        d_log_py: (M, D)

    Returns:
        kernel matrix: (N, M)
    """
    N, D = x.shape
    M = y.shape[0]

    batch_kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=l)
    grad_x_K_fn = jax.grad(batch_kernel.apply, argnums=0)
    vec_grad_x_K_fn = jax.vmap(grad_x_K_fn, in_axes=(0, 0), out_axes=0)
    grad_y_K_fn = jax.grad(batch_kernel.apply, argnums=1)
    vec_grad_y_K_fn = jax.vmap(grad_y_K_fn, in_axes=(0, 0), out_axes=0)

    grad_xy_K_fn = jax.jacfwd(jax.jacrev(batch_kernel.apply, argnums=1), argnums=0)

    def diag_sum_grad_xy_K_fn(x, y):
        return jnp.diag(grad_xy_K_fn(x, y)).sum()

    vec_grad_xy_K_fn = jax.vmap(diag_sum_grad_xy_K_fn, in_axes=(0, 0), out_axes=0)

    x_dummy = jnp.stack([x] * N, axis=1).reshape(N * M, D)
    y_dummy = jnp.stack([y] * M, axis=0).reshape(N * M, D)

    K = batch_kernel.matrix(x, y)
    dx_K = vec_grad_x_K_fn(x_dummy, y_dummy).reshape(N, M, D)
    dy_K = vec_grad_y_K_fn(x_dummy, y_dummy).reshape(N, M, D)
    dxdy_K = vec_grad_xy_K_fn(x_dummy, y_dummy).reshape(N, M)

    part1 = d_log_px @ d_log_py.T * K
    part2 = (d_log_py[None, :] * dx_K).sum(-1)
    part3 = (d_log_px[:, None, :] * dy_K).sum(-1)
    part4 = dxdy_K

    return part1 + part2 + part3 + part4


def stein_Gaussian(x, y, l, d_log_px, d_log_py):
    """
    Stein Gaussian kernel.

    Args:
        x: (N, D)
        y: (M, D)
        l: scalar
        d_log_px: (N, D)
        d_log_py: (M, D)
        
    Returns:
        kernel matrix: (N, M)
    """
    N, D = x.shape
    M = y.shape[0]

    batch_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1., length_scale=l)
    grad_x_K_fn = jax.grad(batch_kernel.apply, argnums=0)
    vec_grad_x_K_fn = jax.vmap(grad_x_K_fn, in_axes=(0, 0), out_axes=0)
    grad_y_K_fn = jax.grad(batch_kernel.apply, argnums=1)
    vec_grad_y_K_fn = jax.vmap(grad_y_K_fn, in_axes=(0, 0), out_axes=0)

    grad_xy_K_fn = jax.jacfwd(jax.jacrev(batch_kernel.apply, argnums=1), argnums=0)

    def diag_sum_grad_xy_K_fn(x, y):
        return jnp.diag(grad_xy_K_fn(x, y)).sum()

    vec_grad_xy_K_fn = jax.vmap(diag_sum_grad_xy_K_fn, in_axes=(0, 0), out_axes=0)

    x_dummy = jnp.stack([x] * N, axis=1).reshape(N * M, D)
    y_dummy = jnp.stack([y] * M, axis=0).reshape(N * M, D)

    K = batch_kernel.matrix(x, y)
    dx_K = vec_grad_x_K_fn(x_dummy, y_dummy).reshape(N, M, D)
    dy_K = vec_grad_y_K_fn(x_dummy, y_dummy).reshape(N, M, D)
    dxdy_K = vec_grad_xy_K_fn(x_dummy, y_dummy).reshape(N, M)

    part1 = d_log_px @ d_log_py.T * K
    part2 = (d_log_py[None, :] * dx_K).sum(-1)
    part3 = (d_log_px[:, None, :] * dy_K).sum(-1)
    part4 = dxdy_K

    return part1 + part2 + part3 + part4


@jax.jit
def my_Matern(x, y, l):
    """
    Matern kernel.

    Args:
        x: (N, D)
        y: (M, D)
        l: scalar

    Returns:
        kernel matrix: (N, M)
    """
    kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=l)
    K = kernel.matrix(x, y)
    return K


# @jax.jit
def dx_Matern(x, y, l):
    """
    Matern kernel derivative with respect to the first input.

    Args:
        x: (N, D)
        y: (M, D)
        l: scalar

    Returns:
        derivative of kernel matrix: (N, M, D)
    """
    N, D = x.shape
    M = y.shape[0]
    kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=l)
    grad_x_K_fn = jax.grad(kernel.apply, argnums=0)
    vec_grad_x_K_fn = jax.vmap(grad_x_K_fn, in_axes=(0, 0), out_axes=0)
    x_dummy = jnp.stack([x] * N, axis=1).reshape(N * M, D)
    y_dummy = jnp.stack([y] * M, axis=0).reshape(N * M, D)
    dx_K = vec_grad_x_K_fn(x_dummy, y_dummy).reshape(N, M, D)
    return dx_K


# @jax.jit
def dy_Matern(x, y, l):
    """
    Matern kernel derivative with respect to the second input.

    Args:
        x: (N, D)
        y: (M, D)
        l: scalar

    Returns:
        derivative of kernel matrix: (N, M, D)
    """
    N, D = x.shape
    M = y.shape[0]
    kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=l)
    grad_y_K_fn = jax.grad(kernel.apply, argnums=1)
    vec_grad_y_K_fn = jax.vmap(grad_y_K_fn, in_axes=(0, 0), out_axes=0)
    x_dummy = jnp.stack([x] * N, axis=1).reshape(N * M, D)
    y_dummy = jnp.stack([y] * M, axis=0).reshape(N * M, D)
    dy_K = vec_grad_y_K_fn(x_dummy, y_dummy).reshape(N, M, D)
    return dy_K


# @jax.jit
def dxdy_Matern(x, y, l):
    """
    The inner product of dx_Matern and dy_Matern
    A fully vecotrized implementation.

    Args:
        x: (N, D)
        y: (M, D)
        l: scalar

    Returns:
        inner product (N, M)
    """
    N, D = x.shape
    M = y.shape[0]

    kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=l)
    grad_xy_K_fn = jax.jacfwd(jax.jacrev(kernel.apply, argnums=1), argnums=0)

    def diag_sum_grad_xy_K_fn(x, y):
        return jnp.diag(grad_xy_K_fn(x, y)).sum()

    vec_grad_xy_K_fn = jax.vmap(diag_sum_grad_xy_K_fn, in_axes=(0, 0), out_axes=0)
    x_dummy = jnp.stack([x] * N, axis=1).reshape(N * M, D)
    y_dummy = jnp.stack([y] * M, axis=0).reshape(N * M, D)
    dxdy_K = vec_grad_xy_K_fn(x_dummy, y_dummy).reshape(N, M)
    return dxdy_K


@jax.jit
def my_RBF(x, y, l):
    """
    RBF kernel.

    Args:
        x: (N, D)
        y: (M, D)
        l: scalar

    Returns:
        kernel matrix: (N, M)
    """
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1., length_scale=l)
    K = kernel.matrix(x, y)
    return K


def jax_dist(x, y):
    return jnp.sqrt(((x - y) ** 2).sum(-1)).squeeze()


distance = jax.vmap(jax_dist, in_axes=(None, 0), out_axes=1)
sign_func = jax.vmap(jnp.greater, in_axes=(None, 0), out_axes=1)


def my_Laplace(x, y, l):
    """
    Laplace kernel.

    Args:
        x: (N, D)
        y: (M, D)
        l: scalar

    Returns:
        kernel matrix: (N, M)
    """
    r = distance(x, y).squeeze()
    return jnp.exp(- r / l)


def dx_Laplace(x, y, l):
    sign = sign_func(x, y).squeeze().astype(float) * 2 - 1
    r = distance(x, y).squeeze()
    part1 = jnp.exp(- r / l) * (-sign)
    return part1


def dy_Laplace(x, y, l):
    sign = sign_func(x, y).squeeze().astype(float) * 2 - 1
    r = distance(x, y).squeeze()
    part1 = jnp.exp(- r / l) * sign
    return part1


def dxdy_Laplace(x, y, l):
    r = distance(x, y).squeeze()
    part1 = jnp.exp(- r / l) * (-1)
    return part1


@jax.jit
def kme_Matern_Gaussian(l, y):
    """
    The implementation of the kernel mean embedding of the Matern kernel with Gaussian distribution
    Only in one dimension, and the Gaussian distribution is N(0, 1)
    
    Args:
        y: (M, D)
        l: scalar

    Returns:
        kernel mean embedding: (M, )
    """
    E10 = 1 - jnp.sqrt(3) * y / l
    E11 = jnp.sqrt(3) / l
    muA = -jnp.sqrt(3) / l

    part11 = jnp.exp((3 + 2 * jnp.sqrt(3) * y * l) / (2 * l ** 2))
    part12 = (E10 + E11 * muA) * norm.cdf(muA - y)
    part13 = E11 / jnp.sqrt(2 * jnp.pi) * jnp.exp(-(y - muA) ** 2 / 2)
    part1 = part11 * (part12 + part13)

    E20 = 1 + jnp.sqrt(3) * y / l
    E21 = jnp.sqrt(3) / l
    muB = jnp.sqrt(3) / l

    part21 = jnp.exp((3 - 2 * jnp.sqrt(3) * y * l) / (2 * l ** 2))
    part22 = (E20 - E21 * muB) * norm.cdf(y - muB)
    part23 = E21 / jnp.sqrt(2 * jnp.pi) * jnp.exp(-(y - muB) ** 2 / 2)
    part2 = part21 * (part22 + part23)

    final = part1 + part2
    pause = True
    return final


@jax.jit
def kme_RBF_Gaussian(mu, Sigma, l, y):
    """
    The implementation of the kernel mean embedding of the RBF kernel with Gaussian distribution
    A fully vectorized implementation.

    Args:
        mu: Gaussian mean, (D, )
        Sigma: Gaussian covariance, (D, D)
        y: (M, D)
        l: scalar

    Returns:
        kernel mean embedding: (M, )
    """
    kme_RBF_Gaussian_func_ = partial(kme_RBF_Gaussian_func, mu, Sigma, l)
    kme_RBF_Gaussian_vmap_func = jax.vmap(kme_RBF_Gaussian_func_)
    return kme_RBF_Gaussian_vmap_func(y)


@jax.jit
def kme_RBF_Gaussian_func(mu, Sigma, l, y):
    """
    The implementation of the kernel mean embedding of the RBF kernel with Gaussian distribution.
    Not vectorized.

    Args:
        mu: Gaussian mean, (D, )
        Sigma: Gaussian covariance, (D, D)
        y: (D, )
        l: scalar

    Returns:
        kernel mean embedding: scalar
    """
    D = mu.shape[0]
    l_ = l ** 2
    Lambda = jnp.eye(D) * l_
    Lambda_inv = jnp.eye(D) / l_
    part1 = jnp.linalg.det(jnp.eye(D) + Sigma @ Lambda_inv)
    part2 = jnp.exp(-0.5 * (mu - y).T @ jnp.linalg.inv(Lambda + Sigma) @ (mu - y))
    return part1 ** (-0.5) * part2


@jax.jit
def kme_double_RBF_Gaussian(mu, Sigma, l):
    """
    The implementation of the initial of the RBF kernel with Gaussian distribution.

    Args:
        mu: Gaussian mean, (D, )
        Sigma: Gaussian covariance, (D, D)
        l: scalar

    Returns:
        initial error: scalar
    """
    l_ = l ** 2
    D = mu.shape[0]
    Lambda = jnp.eye(D) * l_
    Lambda_inv = jnp.eye(D) / l_
    part1 = jnp.linalg.det(jnp.eye(D) + Sigma @ Lambda_inv)
    part2 = jnp.linalg.det(jnp.eye(D) + Sigma @ jnp.linalg.inv(Lambda + Sigma))
    return part1 ** (-0.5) * part2 ** (-0.5)


@jax.jit
def log_normal_RBF(x, y, l, d_log_px, d_log_py):
    """
    Log normal RBF kernel.

    Args:
        x: (N, D)
        y: (M, D)
        l: scalar

    Returns:
        kernel matrix: (N, M)
    """
    return my_RBF(jnp.log(x), jnp.log(y), l)


@jax.jit
def kme_log_normal_RBF(y, l, a, b):
    """
    The implementation of the kernel mean embedding of the log RBF kernel with log normal distribution.

    Args:
        y: (M, D)
        a: mean for the log normal distribution, scalar
        b: std for the log normal distribution, scalar
        l: scalar

    Returns:
        kernel mean embedding: (M, )
    """
    part1 = jnp.exp(-(a ** 2 + jnp.log(y) ** 2) / (2 * (b ** 2 + l ** 2)))
    part2 = jnp.power(y, a / (b ** 2 + l ** 2))
    part3 = b * jnp.sqrt(b ** (-2) + l ** (-2))
    return part1 * part2 / part3


@jax.jit
def kme_double_log_normal_RBF(l, a, b):
    """
    The implementation of the initial error of the log RBF kernel with log normal distribution.

    Args:
        a: mean for the log normal distribution, scalar
        b: std for the log normal distribution, scalar
        l: scalar

    Returns:
        initial error: scalar
    """
    dummy = b ** 2 * jnp.sqrt(b ** (-2) + l ** (-2)) * jnp.sqrt(b ** (-2) + 1. / (b ** 2 + l ** 2))
    return 1. / dummy

# def kme_Matern_Gamma(alpha, beta, l, y):
#     """
#     :param alpha: scalar
#     :param beta: scalar
#     :param l: scalar
#     :param y: (N, )
#     :return: (N, )
#     """
#     aprime = alpha + 1
#     bprime = beta + (jnp.sqrt(3.) / (l ** 2))
#     poly_term = (jnp.sqrt(3.) / (l ** 2) * ((beta ** alpha) / (bprime ** aprime)) * alpha) \
#                 + (1. - jnp.sqrt(3) / (l ** 2) * y) * ((beta / bprime) ** alpha)
#     exp_term = jnp.exp(jnp.sqrt(3.) * y / (l ** 2))
#     return poly_term * exp_term

@jax.jit
def nystrom_inv(matrix, eps=1e-6):
    """
    Nystrom approximation of the inverse of a matrix.

    Args:
        matrix: (N, N)
        eps: scalar

    Returns:
        approx_inv: (N, N)
    """
    rng_key = jax.random.PRNGKey(int(time.time()))
    n = matrix.shape[0]
    m = int(n / 2)
    matrix = matrix - eps * jnp.eye(n)
    matrix_mean = jnp.mean(matrix)
    matrix = matrix / matrix_mean  # Scale the matrix to avoid numerical issues

    # Randomly select m columns
    idx = jax.random.choice(rng_key, n, (m, ), replace=False)

    W = matrix[idx, :][:, idx]
    U, s, V = jnp.linalg.svd(W)

    U_recon = jnp.sqrt(m / n) * matrix[:, idx] @ U @ jnp.diag(1. / s)
    S_recon = s * (n / m)

    Sigma_inv = (1. / eps) * jnp.eye(n)
    approx_inv = Sigma_inv - Sigma_inv @ U_recon @ jnp.linalg.inv(jnp.diag(1. / S_recon) + U_recon.T @ Sigma_inv @ U_recon) @ U_recon.T @ Sigma_inv
    approx_inv = approx_inv / matrix_mean  # Don't forget the scaling!
    return approx_inv


def main():
    seed = 0
    rng_key = jax.random.PRNGKey(seed)
    x = jax.random.uniform(rng_key, shape=(3, 2))
    rng_key, _ = jax.random.split(rng_key)
    y = jax.random.uniform(rng_key, shape=(3, 2))
    l = 0.5
    batch_kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=.5)
    K1 = batch_kernel.matrix(x, y)
    K2 = my_Matern(x, y, 0.5)
    print(K1)
    print(K2)

    print("============")
    batch_kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=.5)
    grad_x_K_fn = jax.grad(batch_kernel.apply, argnums=(0,))
    vec_grad_x_K_fn = jax.vmap(grad_x_K_fn, in_axes=(0, 0), out_axes=1)
    grad_y_K_fn = jax.grad(batch_kernel.apply, argnums=(1,))
    vec_grad_y_K_fn = jax.vmap(grad_y_K_fn, in_axes=(0, 0), out_axes=1)

    seed = 0
    rng_key = jax.random.PRNGKey(seed)
    N = 2
    D = 3
    l = 0.5

    rng_key = jax.random.PRNGKey(seed)
    x = jax.random.uniform(rng_key, shape=(N, D))
    rng_key, _ = jax.random.split(rng_key)
    y = jax.random.uniform(rng_key, shape=(N, D))

    x_dummy = jnp.stack((x, x), axis=0).reshape(N * N, D)
    y_dummy = jnp.stack((y, y), axis=1).reshape(N * N, D)

    dx_K = vec_grad_x_K_fn(x_dummy, y_dummy)[0].reshape(N, N, D)
    dy_K = vec_grad_y_K_fn(x_dummy, y_dummy)[0].reshape(N, N, D)

    print(dx_K)
    print(dy_K)

    print(grad_x_K_fn(x[0, :], y[0, :]))
    print(grad_x_K_fn(x[0, :], y[1, :]))
    print(grad_x_K_fn(x[1, :], y[0, :]))
    print(grad_x_K_fn(x[1, :], y[1, :]))

    print(grad_y_K_fn(x[0, :], y[0, :]))
    print(grad_y_K_fn(x[0, :], y[1, :]))
    print(grad_y_K_fn(x[1, :], y[0, :]))
    print(grad_y_K_fn(x[1, :], y[1, :]))


if __name__ == '__main__':
    main()