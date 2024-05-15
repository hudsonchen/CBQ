import jax
import jax.numpy as jnp
import time
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import pickle

def Geometric_Brownian(n, dt, rng_key, sigma=0.3, S0=1):
    rng_key, _ = jax.random.split(rng_key)
    dWt = jax.random.normal(rng_key, shape=(int(n),)) * jnp.sqrt(dt)
    dlnSt = sigma * dWt - 0.5 * (sigma ** 2) * dt
    St = jnp.exp(jnp.cumsum(dlnSt) + jnp.log(S0))
    return St


def callBS(t, s, K, sigma):
    part_one = jnp.log(s / K) / (sigma * jnp.sqrt(t)) + 0.5 * sigma * jnp.sqrt(t)
    part_two = jnp.log(s / K) / (sigma * jnp.sqrt(t)) - 0.5 * sigma * jnp.sqrt(t)
    return s * norm.cdf(part_one) - K * norm.cdf(part_two)


def BSM_butterfly_analytic():
    seed = int(time.time())
    # seed = 0
    rng_key = jax.random.PRNGKey(seed)
    rng_key, _ = jax.random.split(rng_key)
    K1 = 50
    K2 = 150
    s = -0.2
    t = 1
    T = 2
    sigma = 0.3
    S0 = 50

    # St = Geometric_Brownian(100, t / 100, sigma=sigma, S0=S0)[-1]

    epsilon = jax.random.normal(rng_key)
    St = S0 * jnp.exp(sigma * jnp.sqrt(t) * epsilon - 0.5 * (sigma ** 2) * t)
    psiST_St_1 = callBS(T - t, St, K1, sigma) + callBS(T - t, St, K2, sigma) \
                 - 2 * callBS(T - t, St, (K1 + K2) / 2, sigma)
    psiST_St_2 = callBS(T - t, (1 + s) * St, K1, sigma) + callBS(T - t, (1 + s) * St, K2, sigma) \
                 - 2 * callBS(T - t, (1 + s) * St, (K1 + K2) / 2, sigma)
    L_inner = psiST_St_1 - psiST_St_2
    # print(max(L_inner, 0))
    print(L_inner)

    # Verifies the result from BSM agree with standard Monte Carlo
    # iter_num = 10000
    # L_MC = 0
    # for i in range(iter_num):
    #     epsilon = np.random.normal(0, 1)
    #     ST = St * np.exp(sigma * np.sqrt((T-t)) * epsilon - 0.5 * (sigma ** 2) * (T-t))
    #     psi_ST_1 = max(ST - K1, 0) + max(ST - K2, 0) - 2 * max(ST - (K1+K2) / 2, 0)
    #     psi_ST_2 = max((1+s) * ST - K1, 0) + max((1+s) * ST - K2, 0) - 2 * max((1+s) * ST - (K1+K2) / 2, 0)
    #     L_MC += (psi_ST_1 - psi_ST_2)
    # print(L_MC / iter_num)
    return


def price_visualize(St, N, rng_key, K1=50, K2=150, s=-0.2, sigma=0.3, T=2, t=1):
    """
    :param St: the price St at time t
    :return: The function returns the price ST at time T sampled from the conditional
    distribution p(ST|St), and the loss \psi(ST) - \psi((1+s)ST) due to the shock. Their shape is Nx * Ny
    """
    output_shape = (St.shape[0], N)
    rng_key, _ = jax.random.split(rng_key)
    epsilon = jax.random.normal(rng_key, shape=output_shape)
    ST = St * jnp.exp(sigma * jnp.sqrt((T - t)) * epsilon - 0.5 * (sigma ** 2) * (T - t))
    psi_ST_1 = jnp.maximum(ST - K1, 0) + jnp.maximum(ST - K2, 0) - 2 * jnp.maximum(ST - (K1 + K2) / 2, 0)
    psi_ST_2 = jnp.maximum((1 + s) * ST - K1, 0) + jnp.maximum((1 + s) * ST - K2, 0) - 2 * jnp.maximum(
        (1 + s) * ST - (K1 + K2) / 2, 0)

    loss_list = []
    ST_list = []
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs = axs.flatten()

    St_dummy = St[0]
    log_ST_min = jnp.log(St_dummy) + sigma * jnp.sqrt((T - t)) * (-3) - 0.5 * (sigma ** 2) * (T - t)
    log_ST_max = jnp.log(St_dummy) + sigma * jnp.sqrt((T - t)) * (+3) - 0.5 * (sigma ** 2) * (T - t)
    ST_min = jnp.exp(log_ST_min)
    ST_max = jnp.exp(log_ST_max)
    ST_samples = jnp.linspace(ST_min, ST_max, 100)
    normal_samples = (jnp.log(ST_samples) - jnp.log(St_dummy) + 0.5 * (sigma ** 2) * (T - t)) / sigma
    density = norm.pdf(normal_samples)
    axs[0].plot(ST_samples, density)
    axs[0].set_title(r"The pdf for $p(S_T|S_t)$")
    axs[0].set_xlabel(r"$S_T$")

    for _ in range(1000):
        rng_key, _ = jax.random.split(rng_key)
        epsilon = jax.random.normal(rng_key, shape=(1, 1))
        ST = St_dummy * jnp.exp(sigma * jnp.sqrt((T - t)) * epsilon - 0.5 * (sigma ** 2) * (T - t))
        psi_ST_1 = jnp.maximum(ST - K1, 0) + jnp.maximum(ST - K2, 0) - 2 * jnp.maximum(ST - (K1 + K2) / 2, 0)
        psi_ST_2 = jnp.maximum((1 + s) * ST - K1, 0) + jnp.maximum((1 + s) * ST - K2, 0) - 2 * np.maximum(
            (1 + s) * ST - (K1 + K2) / 2, 0)
        loss_list.append(psi_ST_1 - psi_ST_2)
        ST_list.append(ST)

    ST_dummy = jnp.array(ST_list).squeeze()
    loss_dummy = jnp.array(loss_list).squeeze()
    axs[1].scatter(ST_dummy, loss_dummy)
    axs[1].set_title(r"$\psi(S_T) - \psi((1+s)S_T)$")
    axs[1].set_xlabel(r"$S_T$")
    plt.suptitle(rf"$S_t$ is {St_dummy[0]}")
    plt.savefig(f"./results/finance_dummy.pdf")
    # plt.show()
    return ST, psi_ST_1 - psi_ST_2


@jax.jit
def px_theta_fn(x, theta, x_scale, x_mean, sigma, T_finance, t_finance):
    """
    Computes p(x | theta) for importance sampling.
    p(x | theta) is a log normal distribution with mu=-\sigma^2 / 2 * (T - t) and sigma = \sigma^2 (T - t)
    Args:
        x: shape (N, 1)
        x_scale: float
        x_mean: float
        sigma: float
        T_finance: float
        t_finance: float    
    Returns:
        likelihood: shape (N, 1)
    """
    x_tilde = x * x_scale + x_mean
    z = jnp.log(x_tilde / theta)
    n = (z + sigma ** 2 * (T_finance - t_finance) / 2) / sigma / jnp.sqrt(T_finance - t_finance)
    p_n = jax.scipy.stats.norm.pdf(n)
    p_z = p_n / (sigma * jnp.sqrt(T_finance - t_finance))
    p_x_tilde = p_z / x_tilde
    p_x = p_x_tilde / x_scale
    return p_x


# @jax.jit
def log_px_theta_fn(x, theta, x_scale, x_mean, sigma, T_finance, t_finance):
    """
    Computes log p(x | theta) for importance sampling.
    p(x | theta) is a log normal distribution with mu=-\sigma^2 / 2 * (T - t) and sigma = \sigma^2 (T - t)
    Args:
        x: shape (N, 1)
        x_scale: float
        x_mean: float
        sigma: float
        T_finance: float
        t_finance: float    
    Returns:
        likelihood: shape (N, 1)
    """   
    x_tilde = x * x_scale + x_mean
    z = jnp.log(x_tilde / theta)
    n = (z + sigma ** 2 * (T_finance - t_finance) / 2) / sigma / jnp.sqrt(T_finance - t_finance)
    p_n = jax.scipy.stats.norm.pdf(n)
    p_z = p_n / (sigma * jnp.sqrt(T_finance - t_finance))
    p_x_tilde = p_z / x_tilde
    p_x = p_x_tilde / x_scale
    return jnp.log(p_x).sum()


def scale(Z):
    s = Z.std()
    standardized = Z / s
    return standardized, s


def standardize(Z):
    mean = Z.mean(0)
    std = Z.std(0)
    standardized = (Z - mean) / std
    return standardized, mean, std


def save(T, N, CBQ_mean, CBQ_std, KMS_mean, IS_mean, LSMC_mean,
        time_CBQ, time_IS, time_KMS, time_LSMC, calibration, save_path):
    true_EgX_theta = jnp.load(f"{save_path}/finance_EfX_theta.npy")

    L_CBQ = jnp.maximum(CBQ_mean, 0).mean()
    L_IS = jnp.maximum(IS_mean, 0).mean()
    L_LSMC = jnp.maximum(LSMC_mean, 0).mean()
    L_KMS = jnp.maximum(KMS_mean, 0).mean()
    L_true = jnp.maximum(true_EgX_theta, 0).mean()

    rmse_dict = {}
    rmse_dict['CBQ'] = (L_true - L_CBQ) ** 2
    rmse_dict['IS'] = (L_true - L_IS) ** 2
    rmse_dict['LSMC'] = (L_true - L_LSMC) ** 2
    rmse_dict['KMS'] = (L_true - L_KMS) ** 2
    with open(f"{save_path}/rmse_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(rmse_dict, f)

    time_dict = {'CBQ': time_CBQ, 'IS': time_IS, 'LSMC': time_LSMC, 'KMS': time_KMS}
    with open(f"{save_path}/time_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(time_dict, f)

    jnp.save(f"{save_path}/calibration_T_{T}_N_{N}", calibration)

    methods = ["CBQ", "KMS", "LSMC", "IS"]
    rmse_values = [rmse_dict['CBQ'], rmse_dict['KMS'], rmse_dict['LSMC'], rmse_dict['IS']]
    time_values = [time_dict['CBQ'], time_dict['KMS'], time_dict['LSMC'], time_dict['IS']]

    print("\n\n=======================================")
    print(f"T = {T} and N = {N}")
    print("=======================================")
    print("Methods:    " + " ".join([f"{method:<10}" for method in methods]))
    print("RMSE:       " + " ".join([f"{value:<10.6f}" for value in rmse_values]))
    print("Time (s):   " + " ".join([f"{value:<10.6f}" for value in time_values]))
    print("=======================================\n\n")

    # ============= Debug code =============
    # time_values = [time_CBQ, time_KMS, time_LSMC, time_IS]
    # 
    # print("\n\n=======================================")
    # print(f"T = {T} and N = {N}")
    # print("=======================================")
    # print(" ".join([f"{method:<10}" for method in methods]))
    # print(" ".join([f"{value:<10.6f}" for value in time_values]))
    # print("=======================================\n\n")
    # ============= Debug code =============

    return

def save_large(args, T, N, KMS_mean, LSMC_mean, IS_mean, time_KMS, time_LSMC, time_IS):
    true_EgX_theta = jnp.load(f"{args.save_path}/finance_EgX_theta.npy")

    # Saving this would ethetaplode the memory on cluster
    # jnp.save(f"{args.save_path}/LSMC_mean_T_{T}_N_{N}.npy", LSMC_mean.squeeze())
    # jnp.save(f"{args.save_path}/KMS_mean_T_{T}_N_{N}.npy", KMS_mean)

    L_LSMC = jnp.maximum(LSMC_mean, 0).mean()
    L_KMS = jnp.maximum(KMS_mean, 0).mean()
    L_IS = jnp.maximum(IS_mean, 0).mean()
    L_true = jnp.maximum(true_EgX_theta, 0).mean()

    rmse_dict = {}
    rmse_dict['CBQ'] = None
    rmse_dict['IS'] = (L_true - L_IS) ** 2
    rmse_dict['LSMC'] = (L_true - L_LSMC) ** 2
    rmse_dict['KMS'] = (L_true - L_KMS) ** 2
    with open(f"{args.save_path}/rmse_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(rmse_dict, f)

    time_dict = {'CBQ': None, 'IS': time_IS, 'LSMC': time_LSMC, 'KMS': time_KMS}
    with open(f"{args.save_path}/time_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(time_dict, f)
    pause = True
    return



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
