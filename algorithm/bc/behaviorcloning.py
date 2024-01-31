import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import ast
import numpy as np
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

class BCNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=128, fc2_dims=256, chkpt_dir='./tmp/bc'):
        super(BCNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = 25
        self.checkpoint_file = os.path.join(chkpt_dir, 'behavioral_cloning')
        # Define the layers
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.bn1 = nn.BatchNorm1d(fc1_dims)  # Batch normalization layer
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.BatchNorm1d(fc2_dims)  # Batch normalization layer
        self.fc3 = nn.Linear(fc2_dims, n_actions)   # Output layer  

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-6)

        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.2)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        actions = self.fc3(x)
        # actions = self.max_action * torch.tanh(self.fc3(x))
        return actions

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class BC_Agent:
    def __init__(self, n_actions, input_dims, alpha=0.001, fc1_dims=128, fc2_dims=256, batch_size=64, n_epochs=10):
        self.n_epochs = n_epochs
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.bc = BCNetwork(n_actions, input_dims, alpha, fc1_dims, fc2_dims)

    def take_action(self, observation):
        self.bc.eval()
        with torch.no_grad():
            action = self.bc(observation).cpu().detach().numpy().squeeze()
            # add the gaussian
            action += 0.1  * np.random.randn(self.bc.n_actions)
            # action = np.clip(action, -self.bc.max_action, self.bc.max_action)
            # # random actions...
            # random_actions = np.random.uniform(low=-self.bc.max_action, high=self.bc.max_action,
            #                                 size=self.bc.n_actions)
            # # choose if use the random actions
            # action += np.random.binomial(1, 0.03, 1)[0] * (random_actions - action)
            return action 

    def save_models(self):
        print('... saving models ...')
        self.bc.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.bc.load_checkpoint()

    def sample_data(observations_data, actions_data, batch_size):
        index = np.random.choice(range(observations_data.shape[0]),size=batch_size, replace=False)
        sample_actions = actions_data[index,:]
        sample_observations = observations_data[index,:]
        return sample_observations, sample_actions

    def train_behavioral_cloning(self, observations_csv, actions_csv):
        observations_df = pd.read_csv(observations_csv)
        actions_df = pd.read_csv(actions_csv)

        observations = observations_df['observation'].apply(ast.literal_eval)
        actions = actions_df['action'].apply(ast.literal_eval)

        # scale 把 observation 和 action 都放大1000倍
        # 执行的时候环境的obseration放大1000倍丢给bc，bc输出的action再缩小1000倍 
        observations = torch.tensor(list(observations), dtype=torch.float32, requires_grad=True).to(self.bc.device)*1000
        actions = torch.tensor(list(actions), dtype=torch.float32, requires_grad=True).to(self.bc.device)*1000

        dataloader = DataLoader(TensorDataset(observations, actions), batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()

        for epoch in range(self.n_epochs):
            running_loss = 0.0
            for inputs, targets in dataloader:
                self.bc.optimizer.zero_grad()
                inputs = inputs.to(self.bc.device)
                targets = targets.to(self.bc.device)
                outputs = self.bc(inputs)
                # with torch.set_grad_enabled(True):
                #     outputs = self.bc(inputs)
                #     if isinstance(outputs, Normal):
                #         outputs = outputs.rsample()

                loss = criterion(outputs, targets)
                loss.backward(retain_graph=True)
                self.bc.optimizer.step()
                running_loss += loss.item()
            self.bc.scheduler.step()
            print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {running_loss/len(dataloader):.6f}")
                
            # print(f"Parameters after epoch {epoch+1}:")
            # for name, param in self.bc.named_parameters():
            #     print(f"{name}: {param}")
        return self.bc
