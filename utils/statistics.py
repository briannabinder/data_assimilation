import argparse, h5py, yaml
import numpy as np, ot
# import 

def calculate_w2(states, true_states):

    num_sims, T, ensemble_size, _ = states.shape

    if ensemble_size <= 500:
        N_true_samples = 1000
    elif ensemble_size <= 1000:
        N_true_samples = 2000

    w2_all = []
    for t in range(1, T):
        w2 = 0
        for sim in range(num_sims):
            a = np.ones(N_true_samples) / N_true_samples # Uniform weights for a
            b = np.ones(ensemble_size) / ensemble_size # Uniform weights for b
            M = ot.dist(true_states[sim,t,:N_true_samples], states[sim,t])
            w2 += np.sqrt(ot.emd2(a, b, M))
        w2_all.append(w2/num_sims)

    return np.mean(w2_all)

def calculate_avg_rmse(states, true_state):

    num_sims, T, ensemble_size, _ = states.shape
    
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Postprocess experiment results.")
    parser.add_argument("--exp_id", type=str, help="Experiment ID (e.g., L63_03)")
    parser.add_argument("--filter", type=str, default="KDE", help="Filter name (default: KDE)") # ["KDE_VE", "NNFLOW", KFLOW]
    # parser.add_argument("--ensemble_size", type=int, required=True, help="Number of particles")
    parser.add_argument("--stat", type=str, default="W2", help="Statistic to calculate")
    args = parser.parse_args()

    exp_id, filter_name, statistic = args.exp_id, args.filter, args.stat
    exp_dir = f"./exps/{exp_id}/"

    config_path = exp_dir + "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        model_name = config['model']['name']
        obs_operator_name = config['obs_operator']['name']
        ensemble_sizes = config['sweep_params']['ensemble_sizes']

    # Load truth depending on statistic
    if statistic == "W2":
        truth_path = exp_dir + f"results/{model_name}_{obs_operator_name}_SIR_M100000.h5"
        with h5py.File(truth_path, 'r') as f:
            all_true_updated_states = f['all_updated_states'][:]
    elif statistic == "RMSE":
        data_path = exp_dir + "data.h5"
        with h5py.File(data_path, 'r') as f:
            all_true_states = f['all_true_state'][:]  # Assuming this is the key
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")

    # Prepare output
    save_file_path = exp_dir + f"stats/{statistic}.h5"
    
    with h5py.File(save_file_path, 'w') as file:

        # ---- ENKF ----
        enkf_stats = np.zeros(len(ensemble_sizes))
        for i, ensemble_size in enumerate(ensemble_sizes):
            enkf_path = exp_dir + f"results/{model_name}_{obs_operator_name}_ENKF_M{ensemble_size}.h5"
            with h5py.File(enkf_path, 'r') as f:
                all_enkf_updated_states = f['all_updated_states'][:]
            if statistic == "W2":
                enkf_stats[i] = calculate_w2(all_enkf_updated_states, all_true_updated_states)
            elif statistic == "RMSE":
                enkf_stats[i] = calculate_avg_rmse(all_enkf_updated_states, all_true_states)
        file.create_dataset("ENKF", data=enkf_stats)

        # ---- SIR ----
        sir_stats = np.zeros(len(ensemble_sizes))
        for i, ensemble_size in enumerate(ensemble_sizes):
            sir_path = exp_dir + f"results/{model_name}_{obs_operator_name}_SIR_M{ensemble_size}.h5"
            with h5py.File(sir_path, 'r') as f:
                all_sir_updated_states = f['all_updated_states'][:]
            if statistic == "W2":
                sir_stats[i] = calculate_w2(all_sir_updated_states, all_true_updated_states)
            elif statistic == "RMSE":
                sir_stats[i] = calculate_avg_rmse(all_sir_updated_states, all_true_states)
        file.create_dataset("SIR", data=sir_stats)


        # ---- KDE ----
        if filter_name == "KDE":
            h_ys = config['sweep_params']['kde']['h_ys']
            h_x_mins = config['sweep_params']['kde']['h_x_mins']
            h_x_maxs = config['sweep_params']['kde']['h_x_maxs']

            kde_stats = np.zeros((len(ensemble_sizes), len(h_x_mins), len(h_x_maxs), len(h_ys)))

            for i, ens in enumerate(ensemble_sizes):
                for j, h_x_min in enumerate(h_x_mins):
                    for k, h_x_max in enumerate(h_x_maxs):
                        for l, h_y in enumerate(h_ys):
                            suffix = f"HXMIN{h_x_min}_HXMAX{h_x_max}_HY{h_y}"
                            kde_path = exp_dir + f"results/{model_name}_{obs_operator_name}_KDE_M{ens}_{suffix}.h5"
                            try:
                                with h5py.File(kde_path, 'r') as f:
                                    all_kde_updated_states = f['all_updated_states'][:]
                                if statistic == "W2":
                                    kde_stats[i][j][ k][l] = calculate_w2(all_kde_updated_states, all_true_updated_states)
                                elif statistic == "RMSE":
                                    kde_stats[i][j][k][l] = calculate_avg_rmse(all_kde_updated_states, all_true_states)
                            except FileNotFoundError:
                                print(f"Missing: {kde_path}")
                                kde_stats[i][j][k][l] = np.nan
            file.create_dataset("KDE", data=kde_stats)

        # if filter_name == "KDE":
        #     h_ys = config['sweep_params']['kde']['h_ys']
        #     h_x_mins = config['sweep_params']['kde']['h_x_mins']
        #     h_x_maxs = config['sweep_params']['kde']['h_x_maxs']


        #     # TODO compute_stat
        #     pass
        # else:
        #     pass
    # def calculate statistic for all that filter
    # postprocess(args.exp_id, args.filter, args.ensemble_size)