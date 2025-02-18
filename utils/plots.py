import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display
import os
import numpy as np

def plot_particles(particle_data):

    X = 0
    Y = 1
    Z = 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Create two subplots (XY and XZ)

    fig.suptitle(f"Particle Behavior")
    
    for label, data in particle_data.item():
        # Extract required attributes
        particles = data["particles"]
        color = data.get("color", "black")

        # Extract optional attributes with defaults
        alpha = data.get("alpha", 1)  # Default transparency
        size = data.get("size", 50)  # Default marker size
        marker = data.get("marker", "o")  # Default marker shape

        if particles.shape[0] == 1:
            ax1.scatter(particles[X], particles[Y], c=color, s=size, alpha=alpha, marker=marker, label=label)
            ax2.scatter(particles[X], particles[Z], c=color, s=size, alpha=alpha, marker=marker, label=label)
        else:
            ax1.scatter(particles[:, X], particles[:, Y], c=color, s=size, alpha=alpha, marker=marker, label=label)
            ax2.scatter(particles[:, X], particles[:, Z], c=color, s=size, alpha=alpha, marker=marker, label=label)

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.grid(alpha=0.2)

        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
        ax2.grid(alpha=0.2)

        plt.legend()
        plt.tight_layout()
        plt.show() 

# Plots Lorenz63
def plot_lorenz63(models, filter):

    def timeplot(t_min, ensemble_size, sigma_min_x, sigma_y):

        # Select the model for the current ensemble size
        model = models[(ensemble_size, sigma_min_x, sigma_y)]
        
        # Define the variables to plot: x, y, z
        variables = ['x', 'y', 'z']

        # Create a figure with 3 subplots, one for each variable (x, y, z)
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=300, sharex=True, constrained_layout=True)

        # Loop through each variable (x, y, z)
        for idx, var in enumerate(variables):

            # Extract relevant data for the current variable
            ref_data = model.reference_states[:, idx]
            obs_data = model.observations[:, idx]
            pred_data = model.mean_predictions[:, idx]
            update_data = model.mean_updates[:, idx]
            std_pred = model.std_predictions[:, idx]
            std_upd = model.std_updates[:, idx]

            # Dynamically determine the y-limits
            all_values = np.concatenate([ref_data, obs_data, pred_data, update_data])
            y_min, y_max = np.min(all_values), np.max(all_values)
            margin = 0.1 * (y_max - y_min)  # Add 10% margin to y-limits
            y_min -= margin
            y_max += margin

            ax = axes[idx]  # Get the current axis for the variable

            # Only add labels once (for the shared legend)
            if idx == 0:
                # Add plot, scatter, and error bars for the first variable (reference, observations, prediction, etc.)
                ax.plot(model.reference_times, ref_data, linewidth=1, label='Reference', color='blue')
                ax.scatter(model.reference_times[1:], obs_data, s=10, c='Black', marker='x', zorder=2, label='Observations')
                ax.scatter(model.times, pred_data, s=10, c='Red', zorder=2, label='Prediction mean')
                ax.scatter(model.times, update_data, s=10, c='Green', zorder=2, label='Update mean')
                # ax.errorbar(model.times, pred_data, yerr=std_pred, fmt='none', c="Grey", zorder=0, capsize=3)
                ax.errorbar(model.times, update_data, yerr=std_upd, fmt='none', c="Grey", zorder=0, capsize=3)
                ax.legend()
            else:
                # For subsequent variables, only plot without adding new labels
                ax.plot(model.reference_times, ref_data, linewidth=1, color='blue')
                ax.scatter(model.reference_times[1:], obs_data, s=10, c='Black', marker='x', zorder=2)
                ax.scatter(model.times, pred_data, s=10, c='Red', zorder=2)
                ax.scatter(model.times, update_data, s=10, c='Green', zorder=2)
                # ax.errorbar(model.times, pred_data, yerr=std_pred, fmt='none', c="Grey", zorder=0, capsize=3)
                ax.errorbar(model.times, update_data, yerr=std_upd, fmt='none', c="Grey", zorder=0, capsize=3)

            # Set the y-axis label for each subplot
            ax.set_ylabel(f'{var}(t)')
            ax.set_ylim(y_min, y_max)
            ax.grid(alpha=0.4)

        # Set the x-axis label only for the last subplot (shared x-axis)
        axes[-1].set_xlabel('Time (t)')

        # Set the overall figure title
        plt.suptitle(f"sigma_min_x = {sigma_min_x}, sigma_scale = {filter.sigma_scale}", fontsize=10, color="grey")
        plt.xlim(t_min, t_min + 5)
        
        # Display the plot
        plt.show()

        # Define save function to store the plot as an image file
        def save_plot(button):

            save_dir = "data/timeplots"  # Directory to save the plot
            os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
            
            filename = f"lorenz63_kde_M{ensemble_size}_sigma{sigma_min_x}_{filter.sigma_scale}.png"
            filepath = os.path.join(save_dir, filename)
            fig.savefig(filepath, dpi=300)  # Save the plot
            print(f"Plot saved to {filepath}")

        # Create a save button widget
        save_button = widgets.Button(description="Save Plot", button_style='success')
        save_button.on_click(save_plot)  # Attach the save function to the button click
        display(save_button)  # Display the save button

    # Get available ensemble sizes and set the default
    ensemble_sizes = sorted(set(key[0] for key in models.keys()))
    default_ensemble = ensemble_sizes[0]
    sigma_min_xs = sorted(set(key[1] for key in models.keys()))
    default_sigma_x = sigma_min_xs[0]
    sigma_ys = sorted(set(key[2] for key in models.keys()))
    default_sigma_y = sigma_ys[0]

    # Set time slider limits based on the reference times of the model
    model = models[(default_ensemble, default_sigma_x, default_sigma_y)]
    t_min_start = model.reference_times[0]
    t_min_end = model.reference_times[-1] - 5

    # Create interactive widgets with time slider and ensemble size dropdown
    interact(
        timeplot,
        t_min=widgets.FloatSlider(min=t_min_start, max=t_min_end, step=5, value=t_min_start, description="Start Time"),
        ensemble_size=widgets.Dropdown(options=ensemble_sizes, value=default_ensemble, description="Ensemble"),
        sigma_min_x=widgets.Dropdown(options=sigma_min_xs, value=default_sigma_x, description="\u03C3_x"),
        sigma_y=widgets.Dropdown(options=sigma_ys, value=default_sigma_y, description="\u03C3_y")
    )

def plot_rmses(models, filter):

    ensemble_sizes, sigma_min_xs, sigma_ys = get_keys(models)
    default_sigma_x = sigma_min_xs[0]
    default_sigma_y = sigma_ys[0]

    def plot(sigma_min_x, sigma_y):

        # Calculate the average RMSE for each ensemble size
        avg_rmses = []
        for ensemble_size in ensemble_sizes:
            model = models[(ensemble_size, sigma_min_x, sigma_y)]
            avg_rmse = np.mean(model.rmses)
            avg_rmses.append(avg_rmse)

        # Create scatter plot of average RMSEs vs. ensemble size
        fig = plt.figure(figsize=(8, 5), dpi=300)
        plt.plot(ensemble_sizes, avg_rmses, color='blue', marker='o', label="KDE")

        # Add labels, title, and grid
        plt.xlabel("Ensemble Size")
        plt.ylabel("Average RMSE")
        plt.grid(alpha=0.4)
        plt.title(f"sigma_min_x = {sigma_min_x}, sigma_scale = {filter.sigma_scale}", fontsize=10, c="grey")
        plt.legend()
        plt.show()

        def save_plot(button):

            save_dir = "data/rmses"  # Directory to save the plot
            os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
            
            filename = f"{str(model)}_{str(filter)}_sigma{filter.sigma_config['sigma_min_x']}_{filter.sigma_scale}.png"
            filepath = os.path.join(save_dir, filename)
            fig.savefig(filepath, dpi=300)  # Save the plot
            print(f"Plot saved to {filepath}")

        # Create a save button widget
        save_button = widgets.Button(description="Save Plot", button_style='success')
        save_button.on_click(save_plot)  # Attach the save function to the button click
        display(save_button)  # Display the save button
    
    interact(plot,
             sigma_min_x=widgets.Dropdown(options=sigma_min_xs, value=default_sigma_x, description="\u03C3_x"),
            sigma_y=widgets.Dropdown(options=sigma_ys, value=default_sigma_y, description="\u03C3_y")
             )

def plot_sigmas(models, filters):

    ensemble_sizes, sigma_min_xs, sigma_ys = get_keys(models)

    def plot(ensemble_size):

        pass

def get_keys(models):

    ensemble_sizes = sorted(set(key[0] for key in models.keys()))
    sigma_min_xs = sorted(set(key[1] for key in models.keys()))
    sigma_ys = sorted(set(key[2] for key in models.keys()))

    return ensemble_sizes, sigma_min_xs, sigma_ys

