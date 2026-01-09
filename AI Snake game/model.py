import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.2):
        super(Linear_QNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x

    def save(self, file_name='model.pth'):
        path = os.path.join('model', file_name)
        os.makedirs('model', exist_ok=True)
        torch.save(self.state_dict(), path)

class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

    def train_step(self, state, action, reward, level, next_state, game_over, game_won):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.float)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            game_over = (game_over,)

        pred = self.model(state)
        target = pred.clone()

        with torch.no_grad():
            next_q = self.target_model(next_state).max(1)[0]
        q_new = reward + self.gamma * next_q * (1 - torch.tensor(game_over, dtype=torch.float))

        action_idx = action.argmax(1)
        target.scatter_(1, action_idx.unsqueeze(1), q_new.unsqueeze(1))

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        return loss