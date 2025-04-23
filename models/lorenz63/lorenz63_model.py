from models import BaseModel
import numpy as np
import os
from scipy.integrate import solve_ivp

class Lorenz63(BaseModel):

    def __init__(self, 
                 for_dt, 
                 obs_dt, 
                 T_steps,
                 T_burnin,
                 T_spinup,
                 ensemble_size,
                 sigma,
                 rho,
                 beta,
                 initial_ensemble,
                 initial_time,
                 random_sd,
                 ref_initial_state, # TODO: what if we do not need to generate data
                 obs_op,
                 process_noise = 0
                 ):
        
        super().__init__(initial_ensemble, 
                         initial_time,
                         obs_dt, 
                         T_steps, 
                         ensemble_size, 
                         random_sd)
        
        self.for_dt = for_dt
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.process_noise = process_noise

        self.T_spinup = 2000 # IF this exists make initial time other set init time to 0
        self.T_burnin = T_burnin # IF not exists set to 0
        self.T_spinup = T_spinup

        self.ref_initial_state = ref_initial_state
        self.obs_op = obs_op # NEEDED TO GENERATE DATA AND INITIAL ENSEMBLE if not given
        self._generate_data()
        
        # TODO ref time and times should be the same

    def __str__(self):
        return "L63"
    
    def predict(self, start_states, start_time):
        
        assert self.for_dt < self.obs_dt, "for_dt must be smaller than obs_dt"

        def lorenz63_derivatives(t, state):
            x, y, z = state
            dxdt = self.sigma * (y - x)
            dydt = x * (self.rho - z) - y
            dzdt = x * y - self.beta * z
            return [dxdt, dydt, dzdt]

        t_span = (start_time, start_time + self.obs_dt)
        t_eval = np.arange(start_time, start_time + self.obs_dt, self.for_dt)

        predicted_states = []

        if start_states.ndim == 1: # Makes single state iterable
            start_states = start_states[np.newaxis, :]

        for start_state in start_states:
            solution = solve_ivp(lorenz63_derivatives, t_span, start_state, method='RK45', t_eval=t_eval)
            predicted_state = np.array(solution.y[:,-1])
            predicted_states.append(predicted_state)

        noise = np.random.normal(0, self.process_noise, np.array(predicted_states).shape)
        predicted_states = np.array(predicted_states)

        return predicted_states + noise
    
    def post_process(self):

        # # Define directory and ensure it exists
        # data_dir = "exp/"
        # os.makedirs(data_dir, exist_ok=True)  # Ensure the directory exists

        # # Create a unique filename based on model parameters
        # data_file = f"{data_dir}{str(self)}_T{self.T_steps}_dt{self.obs_dt}.npz"

        # Turns lists into numpy arrays
        self.predicted_states = np.array(self.predicted_states) # SAVE
        self.updated_states = np.array(self.updated_states) # SAVE
        self.times = np.array(self.times) # SAVE

        self.mean_predictions = np.mean(self.predicted_states, axis=1)
        self.mean_updates = np.mean(self.updated_states, axis=1)
        self.std_predictions = np.std(self.predicted_states, axis=1)
        self.std_updates = np.std(self.updated_states, axis=1)

        num_steps = self.T_steps - self.T_burnin

        self.rmses = np.linalg.norm(self.mean_predictions[-num_steps:] - self.reference_states[-num_steps:], axis=1) / np.sqrt(3)
        self.obs_rmses = np.linalg.norm(self.observations[-num_steps:] - self.reference_states[-num_steps:], axis=1) / np.sqrt(3)

    def _generate_data(self):
        """
        Generates reference states and observations for the model if no existing data file is found.
        Saves data as an .npz file and loads it if already available.
        """

        # Define directory and ensure it exists
        data_dir = "inp/"
        os.makedirs(data_dir, exist_ok=True)  # Ensure the directory exists

        # Create a unique filename based on model parameters
        data_file = f"{data_dir}{str(self)}_{str(self.obs_op)}_T{self.T_steps}_dt{self.obs_dt}.npz"

        # If the file exists, load data instead of regenerating
        if os.path.exists(data_file):
            # print(f"Loading existing data from {data_file}")
            data = np.load(data_file)
            self.reference_states = data["reference_states"]
            self.observations = data["observations"]
            self.reference_times = data["reference_times"]
            return

        print(f"Generating new data and saving to {data_file}")

        # Initialize storage lists
        reference_states = [self.ref_initial_state]
        observations = []
        reference_times = [self.initial_time]

        # Generate reference states and observations
        for step in range(self.T_steps):
            time = reference_times[-1]
            prior_state = reference_states[-1]
            # NOTE: For the Lorenz-63 model the prediction model is the forward operator
            next_state = self.predict(prior_state, time)
            observation = self.obs_op.get_observations(next_state[0])

            reference_states.append(next_state[0])
            observations.append(observation)
            reference_times.append(time + self.obs_dt)

        # Convert lists to numpy arrays
        self.reference_states = np.array(reference_states)
        self.observations = np.array(observations)
        self.reference_times = np.array(reference_times)

        # Save the generated data to a compressed NumPy file
        np.savez(data_file, 
                reference_states=self.reference_states, 
                observations=self.observations, 
                reference_times=self.reference_times)

        print(f"Data saved to {data_file}")