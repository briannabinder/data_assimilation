from models import BaseModel
import numpy as np
import os
from scipy.integrate import solve_ivp

class Lorenz96(BaseModel):

    def __init__(self,
                 F,
                 dim,
                 for_dt,
                 obs_op,
                 T_burnin,
                 initial_time,
                 obs_dt, 
                 T_steps, 
                 ensemble_size, 
                 random_sd
                 ):
        
        np.random.seed(random_sd)

        initial_ensemble = np.random.randn(ensemble_size, dim)

        super().__init__(initial_ensemble, 
                         initial_time,
                         obs_dt, 
                         T_steps, 
                         ensemble_size, 
                         random_sd)
        self.F = F
        self.dim = dim
        self.for_dt = for_dt
        self.T_burnin = T_burnin
        
        # Initial data should start with N(0,1) per element of size dim
        self.obs_op = obs_op

        self.ref_initial_state = np.array(np.random.randn(dim))

        self._generate_data()

    def __str__(self):
        return "L96"

    def predict(self, start_states, start_time):

        def lorenz63_derivatives(t, state):
            return (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1) - state + self.F

        t_span = (round(start_time, 10), round(start_time + self.obs_dt, 10))
        num_steps = int(round(self.obs_dt / self.for_dt)) + 1
        t_eval = np.linspace(round(start_time, 10), round(start_time + self.obs_dt, 10), num_steps)
        # print(t_span)
        # print(t_eval)

        predicted_states = []

        if start_states.ndim == 1: # Makes single state iterable
            start_states = start_states[np.newaxis, :]

        for start_state in start_states:
            solution = solve_ivp(lorenz63_derivatives, t_span, start_state, method='RK45', t_eval=t_eval)
            predicted_state = np.array(solution.y[:,-1])
            predicted_states.append(predicted_state)

        return np.array(predicted_states)

    def post_process(self):

        # Turns lists into numpy arrays
        self.predicted_states = np.array(self.predicted_states) # SAVE
        self.updated_states = np.array(self.updated_states) # SAVE
        self.times = np.array(self.times) # SAVE

        self.mean_predictions = np.mean(self.predicted_states, axis=1)
        self.mean_updates = np.mean(self.updated_states, axis=1)
        self.std_predictions = np.std(self.predicted_states, axis=1)
        self.std_updates = np.std(self.updated_states, axis=1)

        num_steps = self.T_steps - self.T_burnin

        self.rmses = np.linalg.norm(self.mean_predictions[-num_steps:] - self.reference_states[-num_steps:], axis=1) / np.sqrt(self.dim)

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
            # NOTE: For the Lorenz-96 model the prediction model is the forward operator
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