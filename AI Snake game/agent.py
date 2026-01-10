import math
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
                self.settings = json.load(file)
        except:
            print('No settings file found!')

        self.n_games = 0
        self.gamma = self.settings['gamma']
        self.memory = deque(maxlen=self.settings['max_memory'])
        self.model = Linear_QNet(self.settings['input'], self.settings['hidden'], self.settings['output'])
        self.target_model = Linear_QNet(self.settings['input'], self.settings['hidden'], self.settings['output'])
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.trainer = QTrainer(self.model, self.target_model, lr=self.settings['lr'], gamma=self.gamma)

        self.epsilon = self.settings['epsilon_start']
        self.epsilon_min = self.settings['epsilon_min']
        self.epsilon_decay = self.settings['epsilon_decay']

        self.target_update_counter = 0
        self.target_update_every = 1000

        # Curriculum
        self.curriculum_level = 1
        self.level_up_threshold = 25
        self.recent_scores = deque(maxlen=50)

    def get_state(self, game):
        head = game.snake[0]
        ray_angles = [(0,-1),(1,0),(0,1),(-1,0),(1,-1),(1,1),(-1,1),(-1,-1)]

        def get_ray_dist(dx, dy):
            dist = 0
            x, y = head.x, head.y
            for i in range(1, max(game.w, game.h) // BLOCK_SIZE + 1):
                x += dx * BLOCK_SIZE
                y += dy * BLOCK_SIZE
                if game.is_collision(Point(x, y)):
                    return 1 / i
            return 0

        dangers = [get_ray_dist(dx, dy) for dx, dy in ray_angles]

        food_dx = (game.food.x - head.x) / game.w
        food_dy = (game.food.y - head.y) / game.h
        special_dx = (game.special_food.x - head.x) / game.w if game.special_food else 0
        special_dy = (game.special_food.y - head.y) / game.h if game.special_food else 0
        has_special = 1 if game.special_food else 0

        special_dist = math.hypot(head.x - game.special_food.x, head.y - game.special_food.y) if game.special_food else float('inf')
        food_dist = math.hypot(head.x - game.food.x, head.y - game.food.y)
        special_priority = 1 if special_dist < food_dist * 1.2 else 0

        state = [
            game.direction == Direction.LEFT,
            game.direction == Direction.RIGHT,
            game.direction == Direction.UP,
            game.direction == Direction.DOWN,
            *dangers,
            game.get_fill_ratio() / 100,
            game.current_level / 5,
            food_dx, food_dy,
            special_dx, special_dy,
            has_special,
            special_priority
        ]
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, level, next_state, game_over, game_won, priority=1.0):
        self.memory.append((state, action, reward, level, next_state, game_over, game_won, priority))

    def train_long_memory(self):
        if len(self.memory) < self.settings['batch_size']:
            return
        priorities = np.array([t[-1] for t in self.memory]) + 1e-5
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), self.settings['batch_size'], p=probs, replace=False)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, levels, next_states, dones, _, _ = zip(*batch)
        self.trainer.train_step(states, actions, rewards, levels, next_states, dones, None)

    def train_short_memory(self, state, action, reward, level, next_state, game_over, game_won):
        loss = self.trainer.train_step(state, action, reward, level, next_state, game_over, game_won)
        return loss.item() if loss is not None else 1.0

    def get_action(self, state):
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.model(state_t)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    agent = Agent()
    game = SnakeGameAI(agent.settings['render_ui'])
    record = 0
    total_score = 0
    step = 0

    while True:
        game.max_levels = agent.curriculum_level
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)
        reward, level, done, score, won, duration = game.play_step(action)
        state_new = agent.get_state(game)

        td_error = agent.train_short_memory(state_old, action, reward, level, state_new, done, won)
        agent.remember(state_old, action, reward, level, state_new, done, won, abs(td_error) + 0.01)

        step += 1
        if step % agent.target_update_every == 0:
            tau = 0.005
            for target_param, param in zip(agent.target_model.parameters(), agent.model.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if done or won:
            reason = game.death_reason
            game.reset(agent.curriculum_level)
            agent.n_games += 1
            agent.train_long_memory()

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games} | Score: {score} | Record: {record} | Level Max: {agent.curriculum_level} | Epsilon: {agent.epsilon:.3f} | Reason: {reason}')

            agent.recent_scores.append(score)
            if len(agent.recent_scores) == 50 and np.mean(agent.recent_scores) > agent.level_up_threshold:
                agent.curriculum_level = min(agent.curriculum_level + 1, 5)
                agent.level_up_threshold += 25
                print(f"*** CURRICULUM ADVANCED TO LEVEL {agent.curriculum_level} ***")

if __name__ == '__main__':
    train()