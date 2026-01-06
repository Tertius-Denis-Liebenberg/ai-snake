import torch
import random
import json
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

class Agent:
    def __init__(self):
        try:
            with open('settings.json', 'r') as file:
                self.default_settings = json.load(file)
        except:
            print('No Settings file found!')

        self.n_games = 0
        self.epsilon = 0 # Randomness
        self.gamma = self.default_settings['gamma'] # Discount Rate
        self.memory = deque(maxlen=self.default_settings['max_memory']) # popleft()
        self.model = Linear_QNet(self.default_settings['input'], self.default_settings['hidden'], self.default_settings['output'])
        self.trainer = QTrainer(self.model, lr=self.default_settings['lr'], gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        has_special_food = game.special_food is not None
        fill_ratio = game.get_fill_ratio()

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y, # food down

            # Has Special Food on the map?
            has_special_food,

            # Special Food Location
            has_special_food and game.special_food.x < game.head.x, # special food left
            has_special_food and game.special_food.x > game.head.x, # special food right
            has_special_food and game.special_food.y < game.head.y, # special food up
            has_special_food and game.special_food.y > game.head.y, # special food down

            # Extra Parameters
            game.current_level,
            fill_ratio
        ]

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, level, next_state, game_over, game_won):
        self.memory.append((state, action, reward, level, next_state, game_over, game_won)) # popleft if max memory is reached

    def train_long_memory(self):
        if len(self.memory) > self.default_settings['batch_size']:
            mini_sample = random.sample(self.memory, self.default_settings['batch_size']) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, levels, next_states, game_overs, game_wons = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, levels, next_states, game_overs, game_wons)

    def train_short_memory(self, state, action, reward, level, next_state, game_over, game_won):
        self.trainer.train_step(state, action, reward, level, next_state, game_over, game_won)

    def get_action(self, state):
        # Random moves: trade off exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_high_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old State
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform the move
        reward, current_level, game_over, score, game_won, duration = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, current_level, state_new, game_over, game_won)

        # remember
        agent.remember(state_old, final_move, reward, current_level, state_new, game_over, game_won)

        # train long memory, plot results
        if game_over or game_won:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(file_name=agent.default_settings['file_name'])

            print(f'|    Game:  {agent.n_games}   |  Score:  {score}  |   High Score:  {record}   |   Duration:  {duration}   |')

            # Plot Results
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot_high_scores.append(record)

            # plot(plot_scores, plot_mean_scores, plot_high_scores)

if __name__ == '__main__':
    train()