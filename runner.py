import os
import sys
import inspect
import numpy as np
# from tqdm.notebook import tqdm 
from tqdm import tqdm

def run_data_assimilation(exp_path, filename, model, filter):

    # Ensure the directory exists
    # exp_path = dir_path
    os.makedirs(exp_path, exist_ok=True)

    # Full file path
    filepath = os.path.join(exp_path, filename)

    if os.path.exists(filepath): # Loads the existing file

        print(f"Loading existing data from {filepath}.")
        data = np.load(filepath, allow_pickle=True)

        model = data['model'].item()
        filter = data['filter'].item()
        
    else: # Runs data assimilation if no file exists

        # Get Initial Ensemble
        current_states = model.initial_ensemble
        current_time = model.initial_time # TODO

        
        if str(filter) == "KDE": # Filter is KDE

            loading_disp_str = f"DA {str(filter)} | M={model.ensemble_size} | {filter.scheduler} \u03C3_x={filter.sigma_min} \u03C3_y={filter.sigma_y}"

        elif str(filter) == "EnKF":

            loading_disp_str = f"DA {str(filter)} | M={model.ensemble_size}"

        # Run Data Assimilation
        for step in range(model.T_steps):
        
            # Prediction Step
            predicted_states = model.predict(current_states, current_time)
            model.add_prediction(predicted_states)

            # Update Step
            observation = model.observations[step]
            updated_states = filter.update(predicted_states, observation)
            model.add_update(updated_states)

            model.add_time()

            current_states = updated_states
            current_time = current_time + model.obs_dt

        model.post_process() 

        # Save the model and filter
        print(f"Saving data to {filepath}.")
        np.savez(filepath, model=model, filter=filter)

    return model