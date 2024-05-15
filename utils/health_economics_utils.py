import jax.numpy as jnp
import pickle
from scipy.stats import norm
import matplotlib.pyplot as plt


def save(args, T, N, rmse_CBQ, rmse_KMS, rmse_LSMC, time_CBQ, time_KMS, time_LSMC, 
         calibration_1, calibration_2):
    methods = ["CBQ", "KMS", "LSMC"]
    rmse_values = [rmse_CBQ, rmse_KMS, rmse_LSMC]

    print("\n\n=======================================")
    print(f"T = {T} and N = {N}")
    print("=======================================")
    print(" ".join([f"{method:<11}" for method in methods]))
    print(" ".join([f"{value:<6.5f}" for value in rmse_values]))
    print("=======================================\n\n")

    rmse_dict = {}
    rmse_dict["CBQ"] = rmse_CBQ
    rmse_dict["KMS"] = rmse_KMS
    rmse_dict["LSMC"] = rmse_LSMC
    with open(f"{args.save_path}/rmse_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(rmse_dict, f)

    time_dict = {}
    time_dict["CBQ"] = time_CBQ
    time_dict["KMS"] = time_KMS
    time_dict["LSMC"] = time_LSMC
    with open(f"{args.save_path}/time_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(time_dict, f)
    
    jnp.save(f"{args.save_path}/calibration_1_T_{T}_N_{N}", calibration_1)
    jnp.save(f"{args.save_path}/calibration_2_T_{T}_N_{N}", calibration_2)
    return


def scale(Z):
    s = Z.std()
    standardized = Z / s
    return standardized, s


def standardize(Z):
    mean = Z.mean(0)
    std = Z.std(0)
    standardized = (Z - mean) / std
    return standardized, mean, std


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
    # plt.xlabel("Credible")
    # plt.ylabel("Coverage")
    # plt.title("Calibration plot")
    # plt.legend()
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.close()
    # plt.show()
    return prediction_interval