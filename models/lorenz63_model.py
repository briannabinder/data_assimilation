import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#TODO:
# - Put reference solutions in the form [[x,y,z],[x,y,z]], [ts] instead of xs, ys, zs, ts
# - Change variable names

class Lorenz63Model:

    def __init__(self, 
                 sigma = 10.0, 
                 rho = 28.0, 
                 beta = 8.0 / 3.0, 
                 initial_condition = [1.508870, -1.531271, 25.46091],
                 end_time = 40,
                 time_step = 0.0001,
                 noise_mean = 0, 
                 noise_variance = 2, 
                 observation_time_step = 0.5,
                 ensemble_size = 2000
                 ):
        
        # Lorenz-63 Numerical Model Parameters
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.init_cond = np.array(initial_condition)
        self.end_time = end_time
        self.dt = time_step

        # Observations Parameters
        self.noise_mean = noise_mean
        self.noise_var = noise_variance
        self.obs_dt = observation_time_step

        # Data Assimilation Parameters
        self.ensemble_size = ensemble_size

        # * * * * * * * * * * 

        self.reference_solution = self._forward(self.init_cond, 0, end_time, dt=time_step, final_only=False)

        self.observations = self._get_observations()

        self.init_states = self.observation_operator(np.tile(self.init_cond, (self.ensemble_size, 1)))

    def predict(self, input_states, t_start, t_end):

        if not isinstance(input_states, np.ndarray):
            print("input state must be a numpy array")
            return
        
        if input_states.ndim == 1:

            return self._forward(input_states, t_start, t_end, dt=self.time_step)
        
        else:

            predicted_states = []
            for input_state in input_states:
                predicted_states.append(self._forward(input_state, t_start, t_end, dt=self.dt))

            return np.array(predicted_states)
        
    def observation_operator(self, states):

        def make_measurement(state):

            x, y, z = state

            noise = np.random.normal(self.noise_mean, np.sqrt(self.noise_var), 3)

            x_obs = x + noise[0]
            y_obs = y + noise[1]
            z_obs = z + noise[2]

            return np.array((x_obs, y_obs, z_obs))
        
        if states.ndim == 1:

            return make_measurement(states)
        
        else:

            observations = []
            for state in states:
                observations.append(make_measurement(state))

            return np.array(observations)

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

    # Pushes the state forward throught the numerical model
    # gets the x, y, z values of the Lorenz model at t_end
    def _forward(self, initial_state, t_start, t_end, dt=0.0001, final_only=True):

        # Computes the derivatives of x, y, z at t
        def get_derivatives(state, t=0):

            x, y, z = state

            dxdt = self.sigma * (y - x)
            dydt = self.rho * x - y - x * z
            dzdt = x * y - self.beta * z

            return [dxdt, dydt, dzdt]

        # A sequence of time points for which to solve for x, y, z
        ts = np.arange(t_start, t_end + dt, dt)

        # Solve system of ODEs
        sol = odeint(get_derivatives, initial_state, ts)

        if final_only == True:
            return sol[-1]
        else:
            return sol, ts

    # Gets the observations 
    def _get_observations(self):
        
        sol, ts = self.reference_solution
        observations, ts_obs = [], []

        for (i, t) in enumerate(ts):
            if t != 0 and t % self.obs_dt == 0:

                obs = self.observation_operator(sol[i])

                observations.append(obs)
                ts_obs.append(t)

        return np.array(observations), np.array(ts_obs)
