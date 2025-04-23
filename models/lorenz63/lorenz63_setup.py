import numpy as np
import argparse
import textwrap
import sys
import scipy.io as sio
import os
from os import walk

from runner import run_data_assimilation
from models.lorenz63 import Lorenz63
from observation_operators import LinearGaussian, CubicGaussian
from filters import KernelDensityEstimation, EnsembleKalmanFilter

def cla():

    formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=50)
    parser = argparse.ArgumentParser(description='list of arguments',formatter_class=formatter)

    parser.add_argument('--ensemble', type=int, default=200, 
                        help=textwrap.dedent('''Number of particles in the ensemble'''))
    parser.add_argument('--seed', type=int, default=123457, 
                        help=textwrap.dedent('''Seed Value'''))
    parser.add_argument('--obs_op_type', type=str, default='linear', choices=['linear', 'cubic'], 
                        help=textwrap.dedent('''Which observation operator to use'''))
    parser.add_argument('--filter_type', type=str, default='KDE', choices=['KDE', 'KDE_hybrid', 'EnKF'],
                        help=textwrap.dedent('''Which filter to use (KDE or EnKF)'''))
    parser.add_argument('--scheduler', type=str, default=None, choices=['VE', 'VP'],
                        help=textwrap.dedent('''Which scheduler to use (VE or VP)'''))
    parser.add_argument('--sigma_min_x', type=float, default=None, 
                        help=textwrap.dedent('''Smallest value of standard deviation in sigma schedule for diffusion'''))
    parser.add_argument('--sigma_max_x', type=float, default=None, 
                        help=textwrap.dedent('''Largest value of standard deviation in sigma schedule for diffusion'''))
    parser.add_argument('--sigma_y', type=float, default=None, 
                        help=textwrap.dedent('''Bandwidth of the Kernel for observations'''))
    
    args = parser.parse_args()

    # Conditional args
    if args.filter_type.startswith("KDE"):

        # Ensures that arguments are provided for KDE
        if args.scheduler is None or args.sigma_min_x is None or args.sigma_max_x is None or args.sigma_y is None:
            parser.error("KDE filter requires --scheduler, --sigma_min_x, --sigma_max_x, and --sigma_y.")

        # Cannot run KDE hybrid if observation operator is not linear
        if args.filter_type == 'KDE_hybrid' and args.obs_op_type != 'linear':
            parser.error("Cannot run KDE hybrid if the observation operator is not linear.")
    
    elif args.filter_type == 'EnKF':

        # Cannot run EnKF if observation operator is not linear
        if args.obs_op_type != 'linear':
            parser.error("Cannot run EnKF if the observation operator is not linear.")

        # Provides a warning that arguments for KDE are not used in EnKF
        if args.scheduler is not None or args.sigma_min_x is not None or args.sigma_max_x is not None or args.sigma_y is not None:
            print("Warning: scheduler, sigma_min_x, sigma_max_x, and sigma_y are ignored with EnKF.", file=sys.stderr)


    return args

def main(PARAMS): # Runs data assimilation for one model and filter

    RANDOM_SD = PARAMS.seed

    # SET UP observation operator
    obs_op_params = {
        'mean': 0,
        'std': 2,
        'random_sd': RANDOM_SD
    }

    if PARAMS.obs_op_type == "linear":
        observation_op = LinearGaussian(**obs_op_params)
    elif PARAMS.obs_op_type == "cubic":
        observation_op = CubicGaussian(**obs_op_params)

    # SET UP model * * * * * * * * * * * * * * * * * * * * * * * * * 
    global_model_params = {
        'sigma': 10.0,
        'rho': 28.0,
        'beta': 8.0 / 3.0,
        'for_dt': 0.05, # Process timestep
        'obs_dt': 0.1, # Observation timestep
        'T_spinup':2000,
        'T_burnin': 2000,
        'T_steps': 4000, # Number of assimilation steps
        'process_noise': 10e-2, # std PROCESS
        'random_sd': RANDOM_SD,
        'initial_time': 200, # TODO: Replace with T_spinup
        'obs_op': observation_op # std # REFERENCE 
    }

    inp_path = "inp/initial/"
    data = sio.loadmat(inp_path + f"spinup_M{PARAMS.ensemble}.mat")['model']
    initial_ensemble = np.array(data['x0'][0][0])
    ref_initial_state = np.array(data['xt'][0][0].T)[2000-1]

    dynamic_model_params = {
        'ensemble_size': PARAMS.ensemble,
        'initial_ensemble': initial_ensemble,
        'ref_initial_state': ref_initial_state
    }

    model_params = {**global_model_params, **dynamic_model_params}
    model = Lorenz63(**model_params)

    # SET UP filter * * * * * * * * * * * * * * * * * * * * * * * * * 
    global_filter_params = {
        'obs_op': observation_op, 
        'random_sd': RANDOM_SD
    } 

    if PARAMS.startswith("KDE"):

        kde_filter_params = {
                        'scheduler': PARAMS.scheduler,
                        'sigma_min': PARAMS.sigma_min_x,
                        'sigma_max': PARAMS.sigma_max_x,
                        'sigma_y': PARAMS.sigma_y,
                        'N_tsteps': 1000  # Pseudo-time steps
                    }
        # TODO add hybrid version as well
        
        filter_params = {**global_filter_params, **kde_filter_params}
        if PARAMS.filter_type == 'KDE_hybrid':
            filter = KernelDensityEstimation(**filter_params, hybrid=True)
        else:
            filter = KernelDensityEstimation(**filter_params)

        filename = f"M{PARAMS.ensemble}_SXmin{PARAMS.sigma_min_x}_SXmax{PARAMS.sigma_max_x}_SY{PARAMS.sigma_y}.npz"

    elif PARAMS.filter_type == 'EnKF':

        filter = EnsembleKalmanFilter(**global_filter_params)

        filename = f"M{PARAMS.ensemble}.npz"

    exp_path = f"exp/{str(observation_op)}/{str(filter)}" 

    run_data_assimilation(exp_path, filename, model, filter)

if __name__ == '__main__':
    PARAMS = cla()
    np.random.seed(PARAMS.seed) # cpu vars
    main(PARAMS)