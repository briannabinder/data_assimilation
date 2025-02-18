# import numpy as np
from utils.logger import get_logger
from tqdm.notebook import tqdm

# TODO: make sure to make all class methods abstract methods

def run_data_assimilation(model, filter):

    logger = get_logger(str(model), str(filter), model.ensemble_size)
    logger.info(f"MODEL={str(model)} | FILTER={str(filter)} | ENSEMBLE SIZE={model.ensemble_size}")

    # Get Initial Ensemble
    # states 
    logger.debug(f"Reference Initial State: {model.ref_initial_state}")

    current_states = model.initial_ensemble
    current_time = model.initial_time

    # Run Data Assimilation
    # for step in range(model.T_steps):
    for step in tqdm(range(model.T_steps), desc=f"DA (M={model.ensemble_size}, \u03C3_x={filter.sigma_config['sigma_min_x']}, \u03C3_y={filter.sigma_config['sigma_y']})", unit="step"):
    
        # Prediction Step
        logger.info(f"PREDICT (step={step + 1})")

        predicted_states = model.predict(current_states, current_time, logger)

        model.add_prediction(predicted_states)

        # Update Step
        logger.info(f"UPDATE (step={step + 1})")

        logger.debug(f"First state in ensemble: {predicted_states[0]}")
        observation = model.observations[step]
        logger.debug(f"Observation: {observation}")

        updated_states = filter.update(predicted_states, observation, logger)

        model.add_update(updated_states)

        model.add_time()
        current_states = updated_states
        current_time = current_time + model.obs_dt

    model.post_process()
    logger.info("FINISHED")