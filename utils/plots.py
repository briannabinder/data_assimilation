import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display

def plot_particles(models):
    """
    Plots ensemble particle behavior interactively with widgets.
    """
    ensemble_sizes, sigma_min_xs, sigma_ys = get_kde_keys(models["KDE VE"])
    default_ensemble = ensemble_sizes[0]
    default_sigma_x = sigma_min_xs[0]
    default_sigma_y = sigma_ys[0]

    # Set the initial model for time limits
    model = models["KDE VE"][(default_ensemble, default_sigma_x, default_sigma_y)]
    t_min_start = (model.T_spinup + model.T_burnin) * model.obs_dt
    t_min_end = model.times[-1]

    fig = None  # Global figure for saving

    # time goes from 400 -> 599.9
    def plot(time, ensemble_size, sigma_min_x, sigma_y, kde_type):
        """Generates scatter plots for particle behavior."""
        nonlocal fig

        model = models[kde_type][(ensemble_size, sigma_min_x, sigma_y)]
        enkf_model = models["EnKF"][ensemble_size]

        # print(model.reference_states.shape)
        # Extract relevant data
        reference = model.reference_states # t (4001) x dim (3)
        observation = model.observations # # t (4000) x dim (3)
        predicted_states = model.predicted_states # t (4001) x M x dim (3) 
        updated_states = model.updated_states # t (4001) x M x dim (3)
        updated_means = model.mean_updates # t (4001) x dim (3) 

        enkf_reference = enkf_model.reference_states # t (4001) x dim (3)
        enkf_observation = enkf_model.observations # # t (4000) x dim (3)
        enkf_predicted_states = enkf_model.predicted_states # t (4001) x M x dim (3) 
        enkf_updated_states = enkf_model.updated_states # t (4001) x M x dim (3)
        enkf_updated_means = enkf_model.mean_updates # t (4001) x dim (3)

        time_idx = int(np.where(np.abs(model.times - time) <= 0.05)[0])

        # STYLE
        input_clr = 'cornflowerblue'
        pred_clr = 'lime'
        upd_clr = 'gold'

        particle_size = 10
        ref_size = 20

        ref_marker = '*'
        obs_marker = 'x'

        # PLOTS
        X, Y, Z = 0, 1, 2
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), dpi=300, sharex=True, sharey=True)
        fig.suptitle(f"Particle Behavior at t={time}")

        ax1.scatter(updated_states[time_idx - 1, :, X], updated_states[time_idx - 1, :, Y], c=input_clr, s=particle_size, alpha=0.7, label=f"Input States")
        ax1.scatter(predicted_states[time_idx, :, X], predicted_states[time_idx, :, Y], c=pred_clr, s=particle_size, alpha=0.7, label=f"Predicted States")
        ax1.scatter(updated_states[time_idx, :, X], updated_states[time_idx, :, Y], c=upd_clr, s=particle_size, alpha=0.7, label=f"Updated States")
        ax1.scatter(updated_means[time_idx][X], updated_means[time_idx][Y], c='red', s=ref_size, label=f"Updated State Mean")
        ax1.scatter(reference[time_idx][X], reference[time_idx][Y], c='blue', marker=ref_marker, s=ref_size, label=f"Reference")
        ax1.scatter(observation[time_idx - 1][X], observation[time_idx - 1][Y], c='black', marker=obs_marker, s=ref_size, label=f"Observation")
        
        ax2.scatter(updated_states[time_idx - 1, :, X], updated_states[time_idx - 1, :, Z], c=input_clr, s=particle_size, alpha=0.7, label=f"Input States")
        ax2.scatter(predicted_states[time_idx, :, X], predicted_states[time_idx, :, Z], c=pred_clr, s=particle_size, alpha=0.7, label=f"Predicted States")
        ax2.scatter(updated_states[time_idx, :, X], updated_states[time_idx, :, Z], c=upd_clr, s=particle_size, alpha=0.7, label=f"Updated States")
        ax2.scatter(updated_means[time_idx][X], updated_means[time_idx][Z], c='red', s=ref_size, label=f"Updated State Mean")
        ax2.scatter(reference[time_idx][X], reference[time_idx][Z], c='blue', marker=ref_marker, s=ref_size, label=f"Reference")
        ax2.scatter(observation[time_idx - 1][X], observation[time_idx - 1][Z], c='black', marker=obs_marker, s=ref_size, label=f"Observation")
        
        ax3.scatter(enkf_updated_states[time_idx - 1, :, X], enkf_updated_states[time_idx - 1, :, Y], c=input_clr, s=particle_size, alpha=0.7, label=f"Input States")
        ax3.scatter(enkf_predicted_states[time_idx, :, X], enkf_predicted_states[time_idx, :, Y], c=pred_clr, s=particle_size, alpha=0.7, label=f"Predicted States")
        ax3.scatter(enkf_updated_states[time_idx, :, X], enkf_updated_states[time_idx, :, Y], c=upd_clr, s=particle_size, alpha=0.7, label=f"Updated States")
        ax3.scatter(enkf_updated_means[time_idx][X], enkf_updated_means[time_idx][Y], c='red', s=ref_size, label=f"Updated State Mean")
        ax3.scatter(enkf_reference[time_idx][X], enkf_reference[time_idx][Y], c='blue', marker=ref_marker, s=ref_size, label=f"Reference")
        ax3.scatter(enkf_observation[time_idx - 1][X], enkf_observation[time_idx - 1][Y], c='black', marker=obs_marker, s=ref_size, label=f"Observation")
        
        ax4.scatter(enkf_updated_states[time_idx - 1, :, X], enkf_updated_states[time_idx - 1, :, Z], c=input_clr, s=particle_size, alpha=0.7, label=f"Input States")
        ax4.scatter(enkf_predicted_states[time_idx, :, X], enkf_predicted_states[time_idx, :, Z], c=pred_clr, s=particle_size, alpha=0.7, label=f"Predicted States")
        ax4.scatter(enkf_updated_states[time_idx, :, X], enkf_updated_states[time_idx, :, Z], c=upd_clr, s=particle_size, alpha=0.7, label=f"Updated States")
        ax4.scatter(enkf_updated_means[time_idx][X], enkf_updated_means[time_idx][Z], c='red', s=ref_size, label=f"Updated State Mean")
        ax4.scatter(enkf_reference[time_idx][X], enkf_reference[time_idx][Z], c='blue', marker=ref_marker, s=ref_size, label=f"Reference")
        ax4.scatter(enkf_observation[time_idx - 1][X], enkf_observation[time_idx - 1][Z], c='black', marker=obs_marker, s=ref_size, label=f"Observation")

        ax1.set_xlabel("X", fontweight='bold')
        ax1.set_ylabel("Y", fontweight='bold')
        ax1.grid(alpha=0.2)
        # ax1.legend()
        
        ax2.set_xlabel("X", fontweight='bold')
        ax2.set_ylabel("Z", fontweight='bold')
        ax2.grid(alpha=0.2)
        # ax2.legend()

        ax3.set_xlabel("X", fontweight='bold')
        ax3.set_ylabel("Y", fontweight='bold')
        ax3.set_title("EnKF")
        ax3.grid(alpha=0.2)
        # ax3.legend()
        
        ax4.set_xlabel("X", fontweight='bold')
        ax4.set_ylabel("Z", fontweight='bold')
        ax4.set_title("EnKF")
        ax4.grid(alpha=0.2)
        ax4.legend()

        plt.tight_layout()
        plt.show()

    def save_plot(_):
        """Saves the current plot to 'data/particleplots'."""
        if fig is None:
            print("No plot available to save.")
            return

        save_dir = "data/particles"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"L63_{kde_type_widget.value}_M{ensemble_size_widget.value}_SX{sigma_min_x_widget.value}_SY{sigma_y_widget.value}.png"
        filepath = os.path.join(save_dir, filename)

        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filepath}")

    # Interactive widgets
    time_widget = widgets.FloatSlider(min=t_min_start, max=t_min_end, step=5, value=t_min_start, description="Start Time")
    ensemble_size_widget = widgets.Dropdown(options=ensemble_sizes, value=default_ensemble, description="Ensemble")
    sigma_min_x_widget = widgets.Dropdown(options=sigma_min_xs, value=default_sigma_x, description="\u03C3_x")
    sigma_y_widget = widgets.Dropdown(options=sigma_ys, value=default_sigma_y, description="\u03C3_y")
    kde_type_widget = widgets.Dropdown(options=["KDE VE", "KDE VP"], value="KDE VE", description="KDE Type")
    save_button = widgets.Button(description="Save Plot", button_style='success')
    save_button.on_click(save_plot)

    # Display widgets
    interact(
        plot,
        time=time_widget,
        ensemble_size=ensemble_size_widget,
        sigma_min_x=sigma_min_x_widget,
        sigma_y=sigma_y_widget,
        kde_type=kde_type_widget
    )
    
    display(save_button)



def plot_timeplots(models):
    """
    Plots time series of reference states, observations, predictions, and updates for a selected model.
    
    Args:
        models (dict): Dictionary containing trained models, with keys as (ensemble_size, sigma_min_x, sigma_y).

    Provides an interactive widget to select parameters and toggle EnKF positions.
    """

    # Extract available ensemble sizes, sigma_min_x values, and sigma_y values
    ensemble_sizes, sigma_min_xs, sigma_ys = get_kde_keys(models["KDE VE"])  # Use VE for structure
    default_ensemble = ensemble_sizes[0]
    default_sigma_x = sigma_min_xs[0]
    default_sigma_y = sigma_ys[0]

    # Set the initial model for time limits
    model = models["KDE VE"][(default_ensemble, default_sigma_x, default_sigma_y)]
    t_min_start = model.reference_times[0]
    t_min_end = model.reference_times[-1] - 5
    t_default = (model.T_spinup + model.T_burnin) * model.obs_dt

    fig, axes = None, None  # Global figure and axes for saving

    def plot(t_min, ensemble_size, sigma_min_x, sigma_y, kde_type, show_enkf):
        """
        Inner function to generate and display a time series plot for a selected model.
        
        Args:
            t_min (float): Start time for the plot.
            ensemble_size (int): Selected ensemble size.
            sigma_min_x (float): Selected sigma_min_x value.
            sigma_y (float): Selected sigma_y value.
            kde_type (str): Either 'KDE VE' or 'KDE VP'.
            show_enkf (bool): Whether to overlay EnKF positions.
        """

        nonlocal fig, axes  # Allow modification of global variables

        # Select the appropriate model
        model = models[kde_type][(ensemble_size, sigma_min_x, sigma_y)]

        # Define the variables to plot: x, y, z
        variables = ['x', 'y', 'z']

        # Create a figure with 3 subplots, one for each variable
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=300, sharex=True, constrained_layout=True)

        for idx, var in enumerate(variables):
            # Extract relevant data
            ref_data = model.reference_states[:, idx]
            obs_data = model.observations[:, idx]
            pred_data = model.mean_predictions[:, idx]
            update_data = model.mean_updates[:, idx]
            std_upd = model.std_updates[:, idx]

            # Set y-axis limits dynamically
            all_values = np.concatenate([ref_data, obs_data, pred_data, update_data])
            y_min, y_max = np.min(all_values), np.max(all_values)
            margin = 0.1 * (y_max - y_min)
            y_min -= margin
            y_max += margin

            ax = axes[idx]
            ax.plot(model.reference_times, ref_data, linewidth=1, label='Reference', color='blue')
            ax.scatter(model.reference_times[1:], obs_data, s=10, c='Black', marker='x', zorder=2, label='Observations')
            ax.scatter(model.times, pred_data, s=10, c='Red', zorder=2, label='Prediction mean')
            ax.scatter(model.times, update_data, s=10, c='Green', zorder=2, label='Update mean')
            ax.errorbar(model.times, update_data, yerr=std_upd, fmt='none', c="Grey", zorder=0, capsize=3)

            # Overlay EnKF positions if checkbox is checked
            if show_enkf and ensemble_size in models["EnKF"]:
                enkf_model = models["EnKF"][ensemble_size]
                enkf_data = enkf_model.mean_updates[:, idx]
                ax.scatter(enkf_model.times, enkf_data, s=10, c='purple', zorder=3, label='EnKF Mean')

            if idx == 0:
                ax.legend()

            ax.set_ylabel(f'{var}(t)')
            ax.set_ylim(y_min, y_max)
            ax.grid(alpha=0.4)

        axes[-1].set_xlabel('Time (t)')
        plt.xlim(t_min, t_min + 5)
        plt.show()

    def save_plot(_):
        """
        Saves the currently displayed plot to the 'data/timeplots' directory.
        """
        if fig is None:
            print("No plot available to save.")
            return

        save_dir = "data/timeplots"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"L63_{kde_type_widget.value}_M{ensemble_size_widget.value}_SX{sigma_min_x_widget.value}_SY{sigma_y_widget.value}.png"
        filepath = os.path.join(save_dir, filename)

        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filepath}")

    # Interactive widgets
    t_min_widget = widgets.FloatSlider(min=t_min_start, max=t_min_end, step=5, value=t_default, description="Start Time")
    ensemble_size_widget = widgets.Dropdown(options=ensemble_sizes, value=default_ensemble, description="Ensemble")
    sigma_min_x_widget = widgets.Dropdown(options=sigma_min_xs, value=default_sigma_x, description="\u03C3_x")
    sigma_y_widget = widgets.Dropdown(options=sigma_ys, value=default_sigma_y, description="\u03C3_y")
    kde_type_widget = widgets.Dropdown(options=["KDE VE", "KDE VP"], value="KDE VE", description="KDE Type")
    show_enkf_widget = widgets.Checkbox(value=False, description="Show EnKF Positions")
    save_button = widgets.Button(description="Save Plot", button_style='success')
    save_button.on_click(save_plot)

    # Display widgets
    interact(
        plot,
        t_min=t_min_widget,
        ensemble_size=ensemble_size_widget,
        sigma_min_x=sigma_min_x_widget,
        sigma_y=sigma_y_widget,
        kde_type=kde_type_widget,
        show_enkf=show_enkf_widget
    )

    display(save_button)



def plot_rmses(models):
    """
    Plots the average RMSE (Root Mean Square Error) for different ensemble sizes 
    across three types of filters: EnKF, KDE VE, and KDE VP. The function creates 
    an interactive plot where users can select different 
    sigma_x and sigma_y values for KDE VE and KDE VP models.
    
    Args:
        models (dict): Dictionary containing trained model results with RMSE values.
    """

    # Extract different filter models from dictionary
    enkf_models = models["EnKF"]
    kde_ve_models = models["KDE VE"]
    kde_vp_models = models["KDE VP"]

    # Get available ensemble sizes, sigma_min_x values, and sigma_y values from KDE models
    ensemble_sizes, sigma_min_xs, sigma_ys = get_kde_keys(kde_ve_models)

    # Default values for sigma_min_x and sigma_y (used for initial plot display)
    default_sigma_x = sigma_min_xs[0]
    default_sigma_y = sigma_ys[0]

    def plot(sigma_min_x, sigma_y):
        """
        Inner function to generate and display a scatter plot of RMSE vs. ensemble size.
        
        Args:
            sigma_min_x (float): Selected sigma_min_x value.
            sigma_y (float): Selected sigma_y value.
        """

        # Lists to store average RMSEs for each filter type
        enkf_avg_rmses = []
        kde_ve_avg_rmses = []
        kde_vp_avg_rmses = []

        # Compute average RMSE for each ensemble size
        for ensemble_size in ensemble_sizes:
            enkf_avg_rmses.append(np.mean(enkf_models[ensemble_size].rmses))
            kde_ve_avg_rmses.append(np.mean(kde_ve_models[(ensemble_size, sigma_min_x, sigma_y)].rmses))
            kde_vp_avg_rmses.append(np.mean(kde_vp_models[(ensemble_size, sigma_min_x, sigma_y)].rmses))

        # Create the plot
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ensemble_sizes, enkf_avg_rmses, color='blue', marker='o', label="EnKF")
        ax.plot(ensemble_sizes, kde_ve_avg_rmses, color='red', marker='o', label="KDE VE")
        ax.plot(ensemble_sizes, kde_vp_avg_rmses, color='green', marker='o', label="KDE VP")

        # Add labels, title, and grid
        ax.set_xlabel("Ensemble Size")
        ax.set_ylabel("Average RMSE")
        ax.grid(alpha=0.4)
        ax.legend()

        # Show the plot
        plt.show()

        # Save plot to file
        save_dir = "data/rmses"  # Directory to save plots
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

        # Define filename and file path
        filename = f"L63_SX{sigma_min_x}_SY{sigma_y}.png"
        filepath = os.path.join(save_dir, filename)

        # Save the figure, overwriting any existing file
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filepath}")

    # Create interactive widgets for sigma_x and sigma_y selection
    interact(plot,
             sigma_min_x=widgets.Dropdown(options=sigma_min_xs, value=default_sigma_x, description="\u03C3_x"),
             sigma_y=widgets.Dropdown(options=sigma_ys, value=default_sigma_y, description="\u03C3_y")
             )

def plot_sigmas(models):
    """
    
    Args:
        models (dict): Dictionary containing trained models, with keys as (ensemble_size, sigma_min_x, sigma_y).
    """

    # Extract available ensemble sizes, sigma_min_x values, and sigma_y values
    ensemble_sizes, sigma_min_xs, sigma_ys = get_kde_keys(models["KDE VE"])  # Use VE for structure
    default_ensemble = ensemble_sizes[0]
    # default_sigma_x = sigma_min_xs[0]
    # default_sigma_y = sigma_ys[0]

    # Set the initial model for time limits
    # model = models["KDE VE"][(default_ensemble, default_sigma_x, default_sigma_y)]
    # t_min_start = model.reference_times[0]
    # t_min_end = model.reference_times[-1] - 5
    # t_default = (model.T_spinup + model.T_burnin) * model.obs_dt

    fig, axes = None, None  # Global figure and axes for saving

    def plot(ensemble_size, kde_type):
        """
        Inner function to generate and display a surface plot for a selected model.
        
        Args:
            ensemble_size (int): Selected ensemble size.
            kde_type (str): Either 'KDE VE' or 'KDE VP'.
        """

        nonlocal fig, axes  # Allow modification of global variables

        X, Y = np.meshgrid(sigma_min_xs, sigma_ys)
        Z = np.zeros_like(X, dtype=float)

        for i in range(len(sigma_min_xs)):
            for j in range(len(sigma_ys)):
                sigma_x = X[j][i]
                sigma_y = Y[j][i]
                Z[j][i] = np.mean(models[kde_type][(ensemble_size, sigma_x, sigma_y)].rmses)

        # Create the plot
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.pcolormesh(X, Y, Z, cmap='viridis')

        # Add labels, title, and grid
        ax.set_xlabel("Sigma X")
        ax.set_ylabel("Sigma Y")
        ax.set_zlabel("Average RMSE")
        ax.set_title(f"RMSE Surface Plot ({kde_type}, Ensemble Size {ensemble_size})", fontweight="bold")

        fig.colorbar(ax, shrink=0.5, aspect=10)

        # Show the plot
        plt.show()

    # def save_plot(_):
    #     """
    #     Saves the currently displayed plot to the 'data/timeplots' directory.
    #     """
    #     if fig is None:
    #         print("No plot available to save.")
    #         return

    #     save_dir = "data/timeplots"
    #     os.makedirs(save_dir, exist_ok=True)

    #     filename = f"L63_{kde_type_widget.value}_M{ensemble_size_widget.value}_SX{sigma_min_x_widget.value}_SY{sigma_y_widget.value}.png"
    #     filepath = os.path.join(save_dir, filename)

    #     fig.savefig(filepath, dpi=300, bbox_inches='tight')
    #     print(f"Plot saved to {filepath}")

    # Interactive widgets
    # t_min_widget = widgets.FloatSlider(min=t_min_start, max=t_min_end, step=5, value=t_default, description="Start Time")
    ensemble_size_widget = widgets.Dropdown(options=ensemble_sizes, value=default_ensemble, description="Ensemble")
    # sigma_min_x_widget = widgets.Dropdown(options=sigma_min_xs, value=default_sigma_x, description="\u03C3_x")
    # sigma_y_widget = widgets.Dropdown(options=sigma_ys, value=default_sigma_y, description="\u03C3_y")
    kde_type_widget = widgets.Dropdown(options=["KDE VE", "KDE VP"], value="KDE VE", description="KDE Type")
    # show_enkf_widget = widgets.Checkbox(value=False, description="Show EnKF Positions")
    # save_button = widgets.Button(description="Save Plot", button_style='success')
    # save_button.on_click(save_plot)

    # Display widgets
    interact(
        plot,
        # t_min=t_min_widget,
        ensemble_size=ensemble_size_widget,
        # sigma_min_x=sigma_min_x_widget,
        # sigma_y=sigma_y_widget,
        kde_type=kde_type_widget,
        # show_enkf=show_enkf_widget
    )

    # display(save_button)


# HELPER FN * * * * * * * * * * * * * * * * * * * * * * * * * * * 

def get_kde_keys(models):
    """
    Extracts unique ensemble sizes, sigma_min_x values, and sigma_y values from a KDE model dictionary.

    Args:
        models (dict): Dictionary of KDE models with tuple keys (ensemble_size, sigma_min_x, sigma_y).

    Returns:
        tuple: (list of ensemble_sizes, list of sigma_min_xs, list of sigma_ys)
    """
    ensemble_sizes = sorted(set(key[0] for key in models.keys()))
    sigma_min_xs = sorted(set(key[1] for key in models.keys()))
    sigma_ys = sorted(set(key[2] for key in models.keys()))

    return ensemble_sizes, sigma_min_xs, sigma_ys
