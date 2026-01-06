import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        # Create intermediate hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            
        # Final hidden layer to output
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))

    def forward(self, x):
        # Apply ReLU to all layers except the last one
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        
        # Last layer (no activation for Q-values)
        x = self.layers[-1](x)
        return x
    
    def save(self, file_name = 'model.pth'):
        model_folder_path = r'.\model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, level, next_state, game_over, game_won):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        level = torch.tensor(level, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            level = torch.unsqueeze(level, 0)
            game_over = (game_over, )
            game_won = (game_won, )

        # Get Predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()