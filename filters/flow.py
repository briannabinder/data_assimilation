import torch, torch.nn as nn, numpy as np
from tqdm import tqdm,trange
from scipy import integrate

def get_activation(activation):
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'SiLU':
        return nn.SiLU()
    elif activation == 'Tanh':
        return nn.Tanh()
    elif activation == 'Sigmoid':
        return nn.Sigmoid()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

class modelNN(nn.Module):
    def __init__(self, n_dim_x=1, n_dim_y=1, width=256, depth=4, activation='ReLU'):
        super(modelNN, self).__init__()
        self.activation = nn.ReLU() if activation == 'ReLU' else nn.SiLU()
        self.n_dim_x = n_dim_x
        self.n_dim_y = n_dim_y
        self.n_dim = n_dim_x + n_dim_y
        self.width = width
        layers = [nn.Linear(self.n_dim + 1, width), self.activation]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(self.activation)
        layers.append(nn.Linear(width, n_dim_x))
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x, t):
        x_inp = torch.cat([x, t], dim=-1)
        return self.fc(x_inp) 
    


class modelNN2(nn.Module):
    def __init__(self, n_dim_x=1, n_dim_y=1, width=256, depth=4, activation='ReLU'):
        super(modelNN2, self).__init__()
        self.activation = nn.ReLU() if activation == 'ReLU' else nn.SiLU()
        self.n_dim_x = n_dim_x
        self.n_dim_y = n_dim_y
        self.n_dim = n_dim_x + n_dim_y
        self.width = width
        layers = [nn.Linear(self.n_dim + 4, width), self.activation]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(self.activation)
        layers.append(nn.Linear(width, n_dim_x))
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x, t):
        t = t.squeeze()
        embed = [t - 0.5, torch.cos(2*np.pi*t), torch.sin(2*np.pi*t), -torch.cos(4*np.pi*t)]
        embed = torch.stack(embed, dim=-1)
        x_inp = torch.cat([x, embed], dim=-1)
        return self.fc(x_inp)        
        

def return_model(model_type):
    if model_type == 'modelNN':
        return modelNN
    elif model_type == 'modelNN2':
        return modelNN2
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

class Flow(nn.Module):
    def __init__(self, n_dim_x=1, n_dim_y=1, width=256, depth=4, activation='ReLU', model_type='modelNN', device='cpu'):
        super().__init__()
        self.net = return_model(model_type)(n_dim_x=n_dim_x, n_dim_y=n_dim_y, width=width, depth=depth, activation=activation).to(device)
        #modelNN2(n_dim_x=n_dim_x, n_dim_y=n_dim_y, width=width, depth=depth, activation=activation).to(device)
        self.loss_func = nn.MSELoss(reduction='mean')
        self.device = device
               
    def forward(self, x, y, t):
        inp = torch.cat([x, y], dim=-1)
        return self.net(inp, t)
    
    def x_t(self, x_0, x_1, t):
        return t* (x_1 - x_0) + x_0
    
    def dot_x_t(self, x_0, x_1, t):
        return x_1 - x_0
    
    def step(self, x_t, y, t_start, t_end):
        t_diff = t_end - t_start
        t_mid = t_start + t_diff / 2
        x_mid = x_t + self(x_t, y, t_start) * t_diff / 2
        return x_t + (t_end - t_start) * self(x_mid, y, t_mid)
    
    def loss(self, x_0, x_1, y, t):
        x_t = self.x_t(x_0, x_1, t)
        dot_x_t = self.dot_x_t(x_0, x_1, t)
        out = self(x_t, y, t)
        return self.loss_func(out, dot_x_t)
    
    def train(self, X, Y, n_iters, batch_size, lr, prior_to_posterior=False, save_path=None):
        
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_history = []
        save_freq = 1000 if n_iters < 100 else 1000
        update_freq = 100 if n_iters < 100 else 1000
        
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        if prior_to_posterior:
            assert batch_size <= int(0.5*X.shape[0]), "Batch size must be less than or equal to half the number of samples in X"
        
        self.net.train()
        # pbar = tqdm(range(n_iters), desc="Loss: ", ncols=100, colour='green')
        for i in range(n_iters):
            self.net.train(mode=True)
            # randomly sample indices for the batch
            # idx = torch.randint(0, X.shape[0], (2*batch_size,)) if prior_to_posterior else torch.randint(0, X.shape[0], (batch_size,))
            # idx_1 = idx[:batch_size] 
            idx_1 = torch.randint(0, X.shape[0], (batch_size,))
            
            x_1 = X[idx_1].to(self.device)
            y = Y[idx_1].to(self.device)
            
            if prior_to_posterior:
                # idx_0 = idx[batch_size:]
                pos_ = torch.randperm(idx_1.shape[0])
                idx_0 = idx_1[pos_]
                x_0 = X[idx_0].to(self.device)
            else:
                x_0 = torch.randn_like(x_1).to(self.device) 
                
            t = torch.rand(batch_size, 1).to(self.device)
            loss = self.loss(x_0, x_1, y, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # if (i+1) % update_freq == 0:
            #     pbar.set_description(f"Loss: {loss.item():.4f}")
                
            loss_history.append(loss.item())
            
            if (i + 1) % save_freq == 0 and save_path is not None:
                save_chkpt = save_path + f"/checkpoints/model_{i+1}.pt"
                torch.save(self.net.state_dict(), save_chkpt)
                save_opt = save_path + f"/checkpoints/optimizer_{i+1}.pt"
                torch.save(optimizer.state_dict(), save_opt)
                
        # delete everything occupying GPU memory
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        # del optimizer, x_0, x_1, y, t, loss, idx_0, idx_1
        return loss_history
    
    def sample(self, x_start, y_cond, n_steps):
        x_path = []
        x_start = x_start.to(self.device)
        y_cond = y_cond.view(1, -1).expand(x_start.shape[0], -1).to(self.device)
        t = torch.linspace(0, 1, n_steps)
        x_t = x_start
        for i in trange(1, t.shape[0]):
            t_prev = t[i-1].view(1, 1).expand(x_t.shape[0], 1).to(self.device)
            t_curr = t[i].view(1, 1).expand(x_t.shape[0], 1).to(self.device)
            x_t = self.step(x_t, y_cond, t_prev, t_curr)
            x_path.append(x_t)
            
            
        # reshape the output to have shape (n_steps, n_samples, n_dim_x)
        x_path = torch.stack(x_path, dim=0).detach().cpu()
        return x_path
    
    def odeint_sampler(self, x_start, y_cond, n_steps, return_path=False):
        self.net.eval()
        x_start = x_start.to(self.device)
        y_cond = y_cond.view(1, -1).expand(x_start.shape[0], -1).to(self.device)
        t_eval = torch.linspace(0, 1, n_steps)
        
        def vel_wrapper(sample, t):
            sample = torch.tensor(sample, device=self.device, dtype=torch.float32).reshape(x_start.shape)
            with torch.no_grad():    
                velocity = self(sample, y_cond, t)
            return velocity.detach().cpu()
        
        def ode_func(t, x):        
            batch_time = torch.ones(x_start.shape[0], 1) * t
            rhs = vel_wrapper(x, batch_time.to(self.device))
            return rhs.numpy().reshape((-1,)).astype(np.float64)
        
        err_tol = 1e-5
        t_eval = np.linspace(0.0, 1.0, n_steps)
        res = integrate.solve_ivp(ode_func, (0.0, 1.0), x_start.reshape(-1).cpu().numpy(), rtol=err_tol, atol=err_tol, method='RK45', dense_output=True, t_eval=t_eval)  
        
        lat_shape = [x_start.shape[0], x_start.shape[1], len(res.t)]
        res_loc = torch.tensor(res.y, device=self.device, dtype=torch.float32).reshape(lat_shape)
        
        final_samples = res_loc[:,:,-1].detach().cpu() #.numpy()
        if return_path:
            res_loc = res_loc.permute(2, 0, 1)
            return res_loc.detach().cpu() #.numpy()
        else:        
            return final_samples
    
    
    
if __name__ == "__main__":
    print("Flow model codes.")
           
            
            
            
            
            
            
            
            
            
        
        