import time, argparse, yaml, os, h5py
from tqdm import tqdm
import numpy as np

from models import Lorenz63, Lorenz96, Toy
from filters import KDE, SKDE, ENKF, SIR

def main():

    start_time = time.time()

    # ==============================================================================
    # === EXPERIMENT SETUP =========================================================
    # ==============================================================================
    print(f"\n--- EXPERIMENT SETUP {'-' * 59}")

    # *=== EXPERIMENT CONFIGURATION ===*

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Data Assimilation Experiment")
    parser.add_argument("--exp_id", required=True, type=str, help="Experiment ID (matches ./exps/<exp_id>/)")

    subparsers = parser.add_subparsers(dest="filter_name", required=True, help="Filter to use")

    # --- KDE Filter ---
    kde_parser = subparsers.add_parser("KDE", help="Kernel Density Estimation filter")
    kde_parser.add_argument("--ensemble_size", required=True, type=int, help="Number of particles")
    kde_parser.add_argument("--h_x_min", required=True, type=float, help="")
    kde_parser.add_argument("--h_y", required=True, type=float, help="")
    kde_parser.add_argument("--scheduler", required=False, default="VE", type=str, help="Scheduler to use")
    kde_parser.add_argument("--h_x_max", required=False, default=50 , type=float, help="")
    kde_parser.add_argument("--N_tsteps", required=False, default=1000, type=int, help="")

    # --- SKDE Filter ---
    skde_parser = subparsers.add_parser("SKDE", help="Stochastic Kernel Density Estimation filter")
    skde_parser.add_argument("--ensemble_size", required=True, type=int, help="Number of particles")
    skde_parser.add_argument("--h_x_min", required=True, type=float, help="")
    skde_parser.add_argument("--h_y", required=True, type=float, help="")
    skde_parser.add_argument("--scheduler", required=False, default="VE", type=str, help="Scheduler to use")
    skde_parser.add_argument("--h_x_max", required=False, default=50 , type=float, help="")
    skde_parser.add_argument("--N_tsteps", required=False, default=5000, type=int, help="")

    # --- SIR Filter ---
    sir_parser = subparsers.add_parser("SIR", help="Sequential Importance Resampling filter")
    sir_parser.add_argument("--ensemble_size", required=True, type=int, help="Number of particles")

    # --- EnKF Filter ---
    enkf_parser = subparsers.add_parser("ENKF", help="Stochastic EnKF filter")
    enkf_parser.add_argument("--ensemble_size", required=True, type=int, help="Number of particles")

    # Parse and unpack arguments
    command_line_args = parser.parse_args()
    command_line_args_dict = vars(command_line_args)

    exp_id      = command_line_args_dict.pop("exp_id")
    filter_name = command_line_args_dict.pop("filter_name")
    filter_args = command_line_args_dict

    # Load YAML config
    config_path = f"./exps/{exp_id}/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        num_sims = config['num_sims']
    
    # *=== MODEL SETUP ===*

    # Get model name and arguments from config
    model_name = config['model']['name']
    model_args = config['model']['args']
    model_args['global_args']['ensemble_size'] = filter_args['ensemble_size'] # Add ensemble size into model args

    # Get observation operator arguments from config
    obs_operator = config['obs_operator']

    if model_name == "Lorenz63":
        model = Lorenz63(model_args, obs_operator)

    elif model_name == "Lorenz96":
        model = Lorenz96(model_args, obs_operator)

    elif model_name == "Toy":
        model = Toy(model_args, obs_operator)

    else:
        raise ValueError(f"\n[ERROR] Unknown model: {model_name}")
    
    # *=== FILTER SETUP ===*

    if filter_name == "KDE":
        filter = KDE(filter_args)
        filter_suffix = f"M{filter_args['ensemble_size']}_HXMIN{filter_args['h_x_min']}_HXMAX{filter_args['h_x_max']}_HY{filter_args['h_y']}_NTSTEPS{filter_args['N_tsteps']}"

    elif filter_name == "SKDE":
        filter = SKDE(filter_args)
        filter_suffix = f"M{filter_args['ensemble_size']}_HXMIN{filter_args['h_x_min']}_HXMAX{filter_args['h_x_max']}_HY{filter_args['h_y']}_NTSTEPS{filter_args['N_tsteps']}"

    elif filter_name == "SIR":
        filter_args['observation_op'] = model.observe
        obs_sigma = config['obs_operator']['args']['sigma']
        if isinstance(obs_sigma, str):
            filter_args['observation_noise'] = eval(config['obs_operator']['args']['sigma'])
        else:
            filter_args['observation_noise'] = config['obs_operator']['args']['sigma']
        filter = SIR(filter_args)
        filter_suffix = f"M{filter_args['ensemble_size']}"

    elif filter_name == "ENKF":
        filter = ENKF(filter_args)
        filter_suffix = f"M{filter_args['ensemble_size']}"

    else:
        raise ValueError(f"\n[ERROR] Unknown filter: {filter_name}")
    
    # *=== EXPERIMENT SUMMARY ===*

    print(f"\n[EXP ID] {exp_id}")

    print(f"\n[MODEL] Name: {model_name}")
    print(f"{' ' * 7} Global Arguments:")
    for key, value in model_args['global_args'].items():
        print(f"{' ' * 7}  * {key} = {value}")
    print(f"{' ' * 7} Local Arguments:")
    for key, value in model_args['local_args'].items():
        print(f"{' ' * 7}  * {key} = {value}")

    print(f"\n[OBS OP] Name: {obs_operator['name']}")
    print(f"{' ' * 8} Arguments:")
    for key, value in obs_operator['args'].items():
        print(f"{' ' * 8}  * {key} = {value}")

    print(f"\n[FILTER] Name: {filter_name}")
    print(f"{' ' * 8} Arguments:")
    for key, value in filter_args.items():
        print(f"{' ' * 8}  * {key} = {value}")

    # ==============================================================================
    # === GENERATE DATA ============================================================
    # ==============================================================================
    print(f"\n--- GENERATE DATA {'-' * 62}")

    data_file_path = f"./exps/{exp_id}/data.h5"

    if os.path.isfile(data_file_path): # Load data from file if it exists

        with h5py.File(data_file_path, 'r') as f:
            times            = f['times'][:]
            all_observations = f['all_observations'][:]
    
    else: # Generate data and save to a file

        print("\n[DATA FILE] Does not exist")

        # Allocate arrays
        all_true_states        = np.zeros((num_sims, model.T_steps + 1, model.state_dim))        # [S x T+1 x dx]
        all_observations       = np.zeros((num_sims, model.T_steps + 1, model.observation_dim))  # [S x T+1 x dy]
        all_observations_clean = np.zeros((num_sims, model.T_steps + 1, model.observation_dim))  # [S x T+1 x dy]

        for sim in range(num_sims): # Generate data for num_sims simulations

            true_states, observations, observations_clean, times = model.generate_data(model.initial_true_state, model.initial_time)

            all_true_states[sim]        = true_states
            all_observations[sim]       = observations
            all_observations_clean[sim] = observations_clean

        # Print summary of array shapes
        print(f"\n[ARR SHAPES] True States: {all_true_states.shape}")
        print(f"{' ' * 12} Observations: {all_observations.shape}")
        print(f"{' ' * 12} Clean Observations: {all_observations_clean.shape}")

        # Save generated data
        with h5py.File(data_file_path, 'w') as f:
            f.create_dataset('times', data=times)
            f.create_dataset('all_true_states', data=all_true_states)
            f.create_dataset('all_observations', data=all_observations)
            f.create_dataset('all_observations_clean', data=all_observations_clean)
    
    print(f"\n[DATA FILE] Location: {data_file_path}")

    # ==============================================================================
    # === RUN DATA ASSIMILATION ====================================================
    # ==============================================================================
    print(f"\n--- RUN DATA ASSIMILATION {'-' * 54}")

    save_dir = f"./exps/{exp_id}/results/{str(filter)}/"
    save_file_path = save_dir + f"{str(model)}_{str(filter)}_{filter_suffix}.h5"
    os.makedir(save_dir, exist_ok=True)

    if not os.path.isfile(save_file_path): # Run data assimilation if results don't already exist

        print("\n[SAVE FILE] Does not exist\n")

        # Allocate arrays
        all_predicted_states       = np.zeros((num_sims, len(times), model.ensemble_size, model.state_dim))        # [S x T+1 x N x dx]
        all_predicted_observations = np.zeros((num_sims, len(times), model.ensemble_size, model.observation_dim))  # [S x T+1 x N x dy]
        all_updated_states         = np.zeros((num_sims, len(times), model.ensemble_size, model.state_dim))        # [S x T+1 x N x dx]


        for sim in range(num_sims): # Run data assimilation for each simulation

            # predicted_states, updated_states = run_data_assimilation(model, filter, all_observations[sim], times)
            predicted_states, predicted_observations, updated_states = run_data_assimilation(model, filter, all_observations[sim], times)

            all_predicted_states[sim]       = predicted_states
            all_predicted_observations[sim] = predicted_observations
            all_updated_states[sim]         = updated_states

            print(f"[STATUS] Simulation {sim+1} complete!")

        # Save output to file
        with h5py.File(save_file_path, 'w') as f:
            f.create_dataset('all_predicted_states', data=all_predicted_states)
            f.create_dataset('all_predicted_observations', data=all_predicted_observations)
            f.create_dataset('all_updated_states', data=all_updated_states)
   
        # Print summary of array shapes
        print(f"\n[ARR SHAPES] Predicted States: {all_predicted_states.shape}")
        print(f"{' ' * 12} Predicted Observations: {all_predicted_observations.shape}")
        print(f"{' ' * 12} Updated States: {all_updated_states.shape}")

    print(f"\n[SAVE FILE] Location: {save_file_path}")

    print(f"\n[TOTAL TIME] {time.time() - start_time} seconds")
    print("Done!")

def run_data_assimilation(model, filter, observations, times):
    """
    Runs one full data assimilation simulation.

    Args:
        model (class):         The model and observation operators
        filter (class):        The filter use in this simulation
        observations (array):  Observations at each time step               -> [N x dy]
        times (array):         Array of times when observations are made    -> [T+1]
    
    Returns:
        predicted_states (array):  The predicted states before each update  -> [T+1 x N x dx]
        updated_states (array):    The updated states after filtering       -> [T+1 x N x dx]
    """

    # Allocate arrays
    predicted_states       = np.zeros((model.T_steps + 1, model.ensemble_size, model.state_dim))        # [T+1 x N x dx]
    predicted_observations = np.zeros((model.T_steps + 1, model.ensemble_size, model.observation_dim))  # [T+1 x N x dy]
    updated_states         = np.zeros((model.T_steps + 1, model.ensemble_size, model.state_dim))        # [T+1 x N x dx]

    # Initialize arrays
    predicted_states[0]       = model.initial_ensemble
    predicted_observations[0] = np.full((model.ensemble_size, model.observation_dim), np.nan)
    updated_states[0]         = predicted_states[0]
    
    for t in range(1, len(times)): # t : [1, T]

        time_start, time_end = times[t-1], times[t]

        # *=== PREDICT STEP ===*
        predicted_states[t] = model.predict(updated_states[t-1], (time_start, time_end))
        predicted_observations[t] = model.observe(predicted_states[t])

        # *=== UPDATE STEP ===*
        updated_states[t] = filter.update(predicted_states[t], predicted_observations[t], observations[t])

    return predicted_states, predicted_observations, updated_states

if __name__ == "__main__":
    main()
    
    # python3 main.py --exp_id {exp_id} KDE --ensemble_size {N} --h_x_min {h_x_min} --h_x_max {h_x_max} --h_y {h_y}
    # python3 main.py --exp_id {exp_id} SIR --ensemble_size {N}
    # python3 main.py --exp_id {exp_id} ENKF --ensemble_size {N}

    # python3 main.py --exp_id L63_01 KDE --ensemble_size 250 --h_x_min 0.4 --h_x_max 50 --h_y 2
    # python3 main.py --exp_id L63_01 SIR --ensemble_size 250 
    # python3 main.py --exp_id L63_01 ENKF --ensemble_size 250
