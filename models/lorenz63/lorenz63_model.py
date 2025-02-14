from models import BaseModel
import numpy as np
from scipy.integrate import solve_ivp

# TODO: External observation operators

class Lorenz63(BaseModel):

    def __init__(self, 
                 for_dt, 
                 obs_dt, 
                 T_steps,
                 T_burnin,
                 ensemble_size,
                 sigma,
                 rho,
                 beta,
                 initial_ensemble,
                 initial_time,
                 rand_seed,
                 ref_initial_state, # TODO: what if we do not need to generate data
                 obs_op,
                 process_noise = 0
                 ):
        
        super().__init__(obs_dt, 
                         T_steps, 
                         ensemble_size)
        
        self.for_dt = for_dt
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.process_noise = process_noise
        np.random.seed(rand_seed)

        self.initial_ensemble = initial_ensemble # Can be given or not
        self.initial_time = initial_time # If not given set to 0 

        self.predicted_states = [initial_ensemble]
        self.updated_states = [initial_ensemble]
        self.times = [initial_time]

        self.T_spinup = 2000 # IF this exists make initial time other set init time to 0
        self.T_burnin = T_burnin # IF not exists set to 0

        self.ref_initial_state = ref_initial_state
        self.obs_op = obs_op # NEEDED TO GENERATE DATA AND INITIAL ENSEMBLE if not given
        self._generate_data()
        
        # TODO ref time and times should be the same

    def __str__(self):
        return "lorenz63"
    
    def add_prediction(self, predicted_states): self.predicted_states.append(predicted_states)

    def add_update(self, updated_states): self.updated_states.append(updated_states)

    def add_time(self): self.times.append(self.times[-1] + self.obs_dt)
    
    def predict(self, start_states, start_time, logger=None):
        
        assert self.for_dt < self.obs_dt, "for_dt must be smaller than obs_dt"

        if logger != None:
            logger.debug(f"Making prediction for time {start_time} -> {start_time + self.obs_dt}")

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

        # Turns lists into numpy arrays
        self.predicted_states = np.array(self.predicted_states)
        self.updated_states = np.array(self.updated_states)
        self.times = np.array(self.times)

        self.mean_predictions = np.mean(self.predicted_states, axis=1)
        self.mean_updates = np.mean(self.updated_states, axis=1)
        self.std_predictions = np.std(self.predicted_states, axis=1)

        num_steps = self.T_steps - self.T_burnin

        self.rmses = np.linalg.norm(self.mean_predictions[-num_steps:] - self.reference_states[-num_steps:], axis=1) / np.sqrt(3)

    def _generate_data(self):

        reference_states = [self.ref_initial_state]
        observations = []
        reference_times = [self.initial_time]

        for step in range(self.T_steps):
            time = self.initial_time + step * self.obs_dt
            prior_state = reference_states[-1]
            next_state = self.predict(prior_state, time, logger=None)
            observation = self.obs_op.get_observations(next_state[0])

            reference_states.append(next_state[0])
            observations.append(observation)
            reference_times.append(time + self.obs_dt)

        self.reference_states = np.array(reference_states)
        self.observations = np.array(observations)
        self.reference_times = np.array(reference_times) # IS THIS NEEDED?
