import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
import sys

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 #learning rate

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9   # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(24, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        point_l_2 = Point(head.x - 40, head.y)
        point_r_2 = Point(head.x + 40, head.y)
        point_u_2 = Point(head.x, head.y - 40)
        point_d_2 = Point(head.x, head.y + 40)

        point_l_3 = Point(head.x - 60, head.y)
        point_r_3 = Point(head.x + 60, head.y)
        point_u_3 = Point(head.x, head.y - 60)
        point_d_3 = Point(head.x, head.y + 60)
        
        point_l_4 = Point(head.x - 80, head.y)
        point_r_4 = Point(head.x + 80, head.y)
        point_u_4 = Point(head.x, head.y - 80)
        point_d_4 = Point(head.x, head.y + 80)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        sc25 = False
        sc50 = False
        sc100 = False
        sc200 = False
        if game.score >= 25:
            sc25 = True
        if game.score >= 50:
            sc50 = True
        if game.score >= 100:
            sc100 = True
        if game.score >= 200:
            sc200 = True
        
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
            
            # Danger straight 2 
            (dir_r and game.is_collision(point_r_2)) or
            (dir_l and game.is_collision(point_l_2)) or
            (dir_u and game.is_collision(point_u_2)) or
            (dir_d and game.is_collision(point_d_2)),
            # Danger right 2
            (dir_u and game.is_collision(point_r_2)) or
            (dir_d and game.is_collision(point_l_2)) or
            (dir_l and game.is_collision(point_u_2)) or
            (dir_r and game.is_collision(point_d_2)),
            # Danger left 2
            (dir_d and game.is_collision(point_r_2)) or
            (dir_u and game.is_collision(point_l_2)) or
            (dir_r and game.is_collision(point_u_2)) or
            (dir_l and game.is_collision(point_d_2)),
            
            # Danger straight 3 
            (dir_r and game.is_collision(point_r_3)) or
            (dir_l and game.is_collision(point_l_3)) or
            (dir_u and game.is_collision(point_u_3)) or
            (dir_d and game.is_collision(point_d_3)),
            # Danger right 3
            (dir_u and game.is_collision(point_r_3)) or
            (dir_d and game.is_collision(point_l_3)) or
            (dir_l and game.is_collision(point_u_3)) or
            (dir_r and game.is_collision(point_d_3)),
            # Danger left 3
            (dir_d and game.is_collision(point_r_3)) or
            (dir_u and game.is_collision(point_l_3)) or
            (dir_r and game.is_collision(point_u_3)) or
            (dir_l and game.is_collision(point_d_3)),
            
            # Danger straight 4
            (dir_r and game.is_collision(point_r_4)) or
            (dir_l and game.is_collision(point_l_4)) or
            (dir_u and game.is_collision(point_u_4)) or
            (dir_d and game.is_collision(point_d_4)),
            # Danger right 4
            (dir_u and game.is_collision(point_r_4)) or
            (dir_d and game.is_collision(point_l_4)) or
            (dir_l and game.is_collision(point_u_4)) or
            (dir_r and game.is_collision(point_d_4)),
            # Danger left 4
            (dir_d and game.is_collision(point_r_4)) or
            (dir_u and game.is_collision(point_l_4)) or
            (dir_r and game.is_collision(point_u_4)) or
            (dir_l and game.is_collision(point_d_4)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y, # food down
            
            # Score more then:
            sc25,   #25
            sc50,   #50
            sc100,  #100
            sc200,  #200
        ]
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
    
    def train_long_memore(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memore(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # random moves: tradeoff exploration/exploitation
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
    games = []
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        
        final_move = agent.get_action(state_old)
        
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memore(state_old, final_move, reward, state_new, done)
        
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # train long memory, plot results
            game.reset()
            agent.n_games += 1
            games.append(agent.n_games)
            agent.train_long_memore()
            
            if score > record:
                record = score
                agent.model.save()
                
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            
            # plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # Display new data
            scores_curve.setData(games, plot_scores)
            mean_scores_curve.setData(games, plot_mean_scores)

            # Set a limit for the y axis (not lower than 0)
            win.setYRange(0, max(max(plot_scores), max(plot_mean_scores)))

            # Add text to the last points on the graph
            if plot_scores:
                # Create a text label for the last point of the curve
                last_score = plot_scores[-1]
                last_mean_score = plot_mean_scores[-1]
                last_game = games[-1]

                # Create text labels
                last_score_text = pg.TextItem('Last Score: {}'.format(last_score), color=(255, 255, 255))
                last_mean_score_text = pg.TextItem('Last Mean Score: {:.2f}'.format(last_mean_score), color=(255, 255, 255))
                
                # Set the text to bold
                last_score_text.setFont(pg.QtGui.QFont("Ebrima", 8, pg.QtGui.QFont.Weight.Bold))
                last_mean_score_text.setFont(pg.QtGui.QFont("Ebrima", 8, pg.QtGui.QFont.Weight.Bold))

                # Set text label positions next to the corresponding curves
                last_score_text.setPos(last_game, last_score)
                last_mean_score_text.setPos(last_game, last_mean_score)
                
                # Remove previous text labels
                for item in win.getPlotItem().items[:]:
                    if isinstance(item, pg.TextItem):
                        win.removeItem(item)

                # Add new text labels to the chart
                win.addItem(last_score_text)
                win.addItem(last_mean_score_text)

            # Refresh the window
            app.processEvents()

if __name__ == "__main__":
    
    app = QApplication([])  # Initializing QGuiApplication

    win = pg.plot(title="Training...")  # Creating a graphic element

    win.setLabel('bottom', 'Number of games')
    win.setLabel('left', 'Score')
    win.setYRange(0, 1)
    # Set the plot background color
    win.setBackground('#262626')
    
    scores_curve = win.plot(pen='r', name='Scores')
    mean_scores_curve = win.plot(pen='g', name='Mean Scores')
    # Set the thickness of the curves
    scores_curve.setPen(pg.mkPen('r', width=3))
    mean_scores_curve.setPen(pg.mkPen('g', width=3))
    
    win.addLegend()
    win.plotItem.legend.addItem(scores_curve, name='Score')
    win.plotItem.legend.addItem(mean_scores_curve, name='Mean Score')
    
    train()