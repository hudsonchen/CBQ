import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import jax
import pickle
import pandas as pd
from tensorflow_probability.substrates import jax as tfp
from scipy.stats import norm
tfd = tfp.distributions

eps = 1e-6


def convert_dict_to_jnp(D):
    D_copy = {}
    for k in D.keys():
        D_copy[k] = jnp.array(D[k])
    return D_copy


def non_zero_ind(A):
    ind = jnp.abs(A) > 2.0
    return ind


def scale(A):
    m = A.mean()
    # m = 10000
    return m, A / m


def standardize(Z):
    mean = Z.mean(0)
    std = Z.std(0)
    # mean = 0.
    # std = 1.
    standardized = (Z - mean) / std
    return standardized, mean, std


@jax.jit
def time_step(beta, gamma, population, St, It, Rt, dt, rng_key):
    dS = -beta * St * It * dt / population
    dI = (beta * St * It / population - gamma * It) * dt
    dR = gamma * It * dt
    St = St + dS
    It = It + dI
    Rt = Rt + dR
    return St, It, Rt


def generate_data(beta, gamma, Time, dt, population, rng_key):
    """
    :param beta: float, infection rate
    :param gamma: float, recovery rate
    :param rng_key:
    :param Time: Time length
    :return: array T*3, the first is number of Susceptible,
    the second is Infected, the third is Recoverdd
    """
    It, Rt = 50., 0.
    St = population - It - Rt

    iter_ = int(Time / dt)

    S_array = jnp.zeros([iter_])
    I_array = jnp.zeros([iter_])
    R_array = jnp.zeros([iter_])

    for i in range(iter_):
        S_array = S_array.at[i].set(St)
        I_array = I_array.at[i].set(It)
        R_array = R_array.at[i].set(Rt)
        St, It, Rt = time_step(beta, gamma, population, St, It, Rt, dt, rng_key)

    # # Plot the data on three separate curves for S(t), I(t) and R(t)
    # fig = plt.figure(facecolor='w')
    # ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    # ax.plot(np.arange(iter_), S_array, 'b', alpha=0.5, lw=2, label='Susceptible')
    # ax.plot(np.arange(iter_), I_array, 'r', alpha=0.5, lw=2, label='Infected')
    # ax.plot(np.arange(iter_), R_array, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    # ax.set_xlabel('Time /days')
    # ax.yaxis.set_tick_params(length=0)
    # ax.xaxis.set_tick_params(length=0)
    # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    # legend = ax.legend()
    # legend.get_frame().set_alpha(0.5)
    # plt.title(f'Infection rate is {beta}')
    # plt.show()

    D = {'S': S_array, 'I': I_array, 'R': R_array}
    pause = True
    return D['I']


def save(args, T, N, CBQ_mean, KMS_mean, LSMC_mean, IS_mean,
         ground_truth_array, CBQ_time, KMS_time, LSMC_time, IS_time, calibration):
    KMS_mean = KMS_mean.squeeze()
    time_dict = {'CBQ': CBQ_time, 'IS': IS_time, 'LSMC': LSMC_time, 'KMS': KMS_time}
    with open(f"{args.save_path}/time_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(time_dict, f)

    rmse_dict = {}
    rmse_dict['CBQ'] = jnp.sqrt(((ground_truth_array - CBQ_mean) ** 2).mean())
    rmse_dict['IS'] = jnp.sqrt(((ground_truth_array - IS_mean) ** 2).mean())
    rmse_dict['LSMC'] = jnp.sqrt(((ground_truth_array - LSMC_mean) ** 2).mean())
    rmse_dict['KMS'] = jnp.sqrt(((ground_truth_array - KMS_mean) ** 2).mean())
    with open(f"{args.save_path}/rmse_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(rmse_dict, f)

    jnp.save(f"{args.save_path}/calibration_T_{T}_N_{N}", calibration)

    methods = ["CBQ", "KMS", "LSMC", "IS"]
    rmse_values = [rmse_dict['CBQ'], rmse_dict['KMS'], rmse_dict['LSMC'], rmse_dict['IS']]

    print("\n\n========================================")
    print(f"T = {T} and N = {N}")
    print("========================================")
    print(" ".join([f"{method:<11}" for method in methods]))
    print(" ".join([f"{value:<6.5f}" for value in rmse_values]))
    print("========================================\n\n")
    
    # ======================================== Debug code ========================================
    # time_values = [time_dict['CBQ'], time_dict['KMS'], time_dict['LSMC'], time_dict['IS']]

    # print("\n\n=======================================")
    # print(f"T = {T} and N = {N}")
    # print("=======================================")
    # print(" ".join([f"{method:<10}" for method in methods]))
    # print(" ".join([f"{value:<10.6f}" for value in time_values]))
    # print("=======================================\n\n")

    # plt.figure()
    # plt.plot(theta_test, CBQ_mean, color='blue', label='CBQ')
    # plt.plot(theta_test, KMS_mean, color='red', label='KMS')
    # plt.plot(theta_test, LSMC_mean, color='green', label='LSMC')
    # plt.plot(theta_test, IS_mean, color='orange', label='IS')
    # plt.plot(theta_test, ground_truth_array, color='black', label='True')
    # plt.scatter(theta_array, CBQ_mean_array, color='orange')
    # plt.fill_between(theta_test, CBQ_mean - CBQ_std, CBQ_mean + CBQ_std, alpha=0.2, color='blue')
    # plt.legend()
    # plt.title(f"T={T}, N={N}")
    # plt.savefig(f"{args.save_path}/figures/SIR_T_{T}_N_{N}.pdf")
    # # plt.show()
    # plt.close()
    # pause = True
    # ======================================== Debug code ========================================
    return


def save_large(args, T, N, KMS_mean, LSMC_mean, IS_mean, ground_truth_array, KMS_time, LSMC_time, IS_time):
    time_dict = {'BMS': None, 'IS': IS_time, 'LSMC': LSMC_time, 'KMS': KMS_time}
    with open(f"{args.save_path}/time_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(time_dict, f)

    rmse_dict = {}
    rmse_dict['IS'] = ((ground_truth_array - IS_mean) ** 2).mean()
    rmse_dict['LSMC'] = ((ground_truth_array - LSMC_mean) ** 2).mean()
    rmse_dict['KMS'] = ((ground_truth_array - KMS_mean) ** 2).mean()
    rmse_dict['BMS'] = None
    with open(f"{args.save_path}/rmse_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(rmse_dict, f)
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
