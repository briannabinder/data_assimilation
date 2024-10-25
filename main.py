# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Ensures the same randomness every run
np.random.seed(1)

class Lorenz63:

    def __init__(self, 
                 sigma=10, rho=28, beta=8/3, 
                 init_cond=[1.508870, -1.531271, 25.46091],
                 forward_dt=0.0001, 
                 noise_mean=0, noise_var=2, obs_timestep=0.5,
                 N_states=3):
        
        # Lorenz Model Parameters
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.init_cond = init_cond
        self.forward_dt = forward_dt

        # Observations Parameters
        self.noise_mean = noise_mean
        self.noise_var = noise_var
        self.obs_timestep = obs_timestep

        # Data Assimilation Parameters
        self.N_states = N_states

        self.ref_solution = self.forward(self.init_cond, 0, 40, dt=self.forward_dt)

        self.observations = self.get_observations()
        self.init_states = self.get_init_states()

    # Prediction model that pushes the state forward
    # gets the x, y, z values of the Lorenz model at t_end
    def forward(self, input_state, t_start, t_end, dt=0.0001, final_only=False):

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
        sol = odeint(get_derivatives, input_state, ts)
        xs, ys, zs = sol.T 

        if final_only == True:
            return xs[-1], ys[-1], zs[-1]
        else:
            return xs, ys, zs, ts

    def obs_operator(self, state):

        x, y, z = state

        noise = np.random.normal(self.noise_mean, np.sqrt(self.noise_var), 3)

        x_obs = x + noise[0]
        y_obs = y + noise[1]
        z_obs = z + noise[2]

        return x_obs, y_obs, z_obs
    
    # * * * * * * * * * * * * * * * 

    # Gets the observations 
    def get_observations(self):
        
        xs, ys, zs, ts = self.ref_solution
        xs_obs, ys_obs, zs_obs, ts_obs = [], [], [], []

        for (i, t) in enumerate(ts):
            if t != 0 and t % self.obs_timestep == 0:

                x_obs, y_obs, z_obs = self.obs_operator((xs[i], ys[i], zs[i]))

                xs_obs.append(x_obs)
                ys_obs.append(y_obs)
                zs_obs.append(z_obs)
                ts_obs.append(t)

        return np.array(xs_obs), np.array(ys_obs), np.array(zs_obs), np.array(ts_obs)

    # Creates the initial state (initial condition + noise)
    def get_init_states(self):

        init_states = []

        for i in range(self.N_states):

            init_states.append(self.obs_operator(self.init_cond))

        return np.array(init_states)

    # Plots x(t)
    def plot(self):

        xs, _, _, ts = self.ref_solution
        xs_obs, _, _, ts_obs = self.observations

        plt.figure(figsize=(10, 5))

        plt.plot(ts, xs, linewidth=1, label='Reference Solution x')
        plt.scatter(ts_obs, xs_obs, s=10, c='Black', zorder=2, label='Observations x')

        plt.xlabel('Time (t)')
        plt.xlim(0, 40)
        plt.ylabel('Value of x(t)')
        plt.ylim(-20, 20)
        plt.legend()
        plt.minorticks_on()
        plt.grid(alpha=0.2)

        plt.show()

    # Prints states for checking
    def print_states(self, time, **states):

        if time <= 0:

            print(f"Time: 0")
            print(f"Reference: [{self.ref_solution[0][0]} {self.ref_solution[1][0]} {self.ref_solution[2][0]}]")
            print("States:")
            for i in range(self.N_states):
                print(f"{i+1}: {self.init_states[i]}")
        
        else:

            print(f"\nTime: {time}")
            ridx = int(time/self.forward_dt)
            print(f"Reference: [{self.ref_solution[0][ridx]} {self.ref_solution[1][ridx]} {self.ref_solution[2][ridx]}]")
            oidx = int(time/self.obs_timestep)
            print(f"Observation: [{self.observations[0][oidx-1]} {self.observations[1][oidx-1]} {self.observations[2][oidx-1]}]")
            print("States:")
            for i in range(self.N_states):
                print(f"{i+1}: Current:     {states['current'][i]}")
                print(f"   Predicted:   {states['predicted'][i]}")
                print(f"   Measurement: {states['measurement'][i]}")

# * * * * * * * * * * * * * * * * * * * * * * * * 


def main(system):

    def data_assimilation(time):

        if time <= 0:

            # Printing states to check
            system.print_states(0)

            return system.init_states
            
        else:

            current_states = data_assimilation(time - system.obs_timestep)

            # PREDICTION STEP

            predicted_states = []

            for i in range(system.N_states):
                predicted_states.append(system.forward(current_states[i], 
                                                       time - L63.obs_timestep, 
                                                       time, 
                                                       final_only=True))
            
            predicted_states = np.array(predicted_states)

            predicted_obs = system.obs_operator(predicted_states)

            # Printing states to check
            system.print_states(time, 
                                current=current_states, 
                                predicted=predicted_states, 
                                measurement=predicted_obs)

            # TODO: UPDATE STEP

            return predicted_states

    data_assimilation(2.0)
    # system.plot()

    return 0

if __name__ == '__main__':

    L63 = Lorenz63()
    main(L63)