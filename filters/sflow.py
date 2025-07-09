from filters import BaseFilter

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from scipy.integrate import solve_ivp
from .flow import Flow

class SFLOW(BaseFilter):

    def __str__(self): return f"SFLOW_{self.scheduler}"

    def __init__(self, filter_args):
        
        # Filter Parameters
        self.ensemble_size = filter_args['ensemble_size']
        self.n_dim_x = filter_args['n_dim_x']
        self.n_dim_y = filter_args['n_dim_y']
        self.width = filter_args['width']
        self.depth = filter_args['depth']
        self.activation = filter_args['activation']
        self.model_type = filter_args['model_type']
        self.device = filter_args['device']
        self.n_iters = filter_args['n_iters']
        self.batch_size = filter_args['batch_size']
        self.lr = filter_args['lr']
        self.prior_to_posterior = filter_args['prior_to_posterior']

    
    def update(self, predicted_states, predicted_observations, observation):

        # Min max normalize the predicted states between -0.5 and 0.5
        scaler_x = MinMaxScaler(feature_range=(-0.5, 0.5))
        train_x = scaler_x.fit_transform(predicted_states)
        
        # Min max normalize the predicted observations between -0.5 and 0.5
        scaler_y = MinMaxScaler(feature_range=(-0.5, 0.5))
        train_y = scaler_y.fit_transform(predicted_observations)
        
        training_data = np.concatenate((train_x, train_y), axis=1)
        training_data = torch.tensor(training_data, dtype=torch.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_dim = training_data.shape[1]
        model = modelNN(n_dim, width=64, depth=4).to(device)

        flow = Flow(n_dim_x=self.n_dim_x, n_dim_y=self.n_dim_y, width=self.width, depth=self.depth, activation=self.activation, model_type=self.model_type, device=self.device)

        training_data_x = training_data[:, :self.n_dim_x] if self.n_dim_x > 1 else training_data[:, :self.n_dim_x].unsqueeze(1)
        training_data_y = training_data[:, self.n_dim_x:] if self.n_dim_y > 1 else training_data[:, self.n_dim_x:].unsqueeze(1)
        # train the model
        loss_history = flow.train(X=training_data_x,
                                Y=training_data_y,
                                n_iters=self.n_iters, batch_size=self.batch_size, lr=self.lr, 
                                prior_to_posterior=self.prior_to_posterior, save_path=None)

        y_cond_transformed = scaler_y.transform(observation.reshape(1, -1))
        y_cond = torch.tensor(y_cond_transformed, dtype=torch.float32).to(device)  
        
        n_samples = train_x.shape[0]

        if prior_to_posterior:
            x_start = training_data_x
        else:
            x_start = torch.randn(sample_batch_size, 1, device=device).float()
        # x_path = flow.sample(x_start=x_start, y_cond=y_cond, n_steps=1000)
        x_path = flow.odeint_sampler(x_start=x_start, y_cond=y_cond, n_steps=100, return_path=True)

        samples_final = x_path[-1].numpy()        
        updated_states = scaler_x.inverse_transform(samples_final.reshape(-1, self.n_dim_x))

        return updated_states

