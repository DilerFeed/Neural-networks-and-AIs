from mss import mss     # for screen cap
import pydirectinput    # sending commands
import cv2              # frame processing
import numpy as np      # transformational framework
import pytesseract      # OCR for game over extraction
from matplotlib import pyplot as plt    # visualize captured frames
import time
# environment components
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

import os
# for saving models
from stable_baselines3.common.callbacks import BaseCallback
# check environment
from stable_baselines3.common import env_checker

# import thee Deep-Q-Network algorithm
from stable_baselines3 import DQN


pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

class WebGame(Env):
    # setup the environment action and observation shapes
    def __init__(self):
        # subclass model
        super().__init__()
        # setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # define extraction parameters for the game
        self.cap = mss()
        self.game_location = {'top':300, 'left':0, 'width':600, 'height':500}
        self.done_location = {'top':405, 'left':630, 'width':660, 'height':70}
        
    def step(self, action):
        # 0 = Spacebar, 1 = down, 2 = no action
        action_map = {
            0:'space',
            1:'down',
            2:'no_op'
        }    
        if action != 2:
            pydirectinput.press(action_map[action])
        # checking whether the game is done
        terminated, truncated, done_cap = self.get_done()
        # get the next observation
        new_observation = self.get_observation()
        # reward (point for every frame we are alive)
        reward = 1
        # indo dict
        info = {}
        
        return new_observation, reward, terminated, truncated, info
    
    def render(self):
        #cv2.imshow('Game', self.get_observation())
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:, :, :3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close
            
    def close(self):
        cv2.destroyAllWindows()
    
    def reset(self, seed=0):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        # indo dict
        info = {}
        
        return self.get_observation(), info
    
    # get the part of the observation that we want
    def get_observation(self):
        # get screen capture of game
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3]  #.astype(np.uint8)
        # grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # resize
        resized = cv2.resize(gray, (100, 83))
        # add channels first
        channel = np.reshape(resized, (1, 83, 100))
        
        return channel
    
    # get the done text using OCR
    def get_done(self):
        # get done screen
        done_cap = np.array(self.cap.grab(self.done_location))[:, :, :3]
        # valid done text
        done_strings = ['GAME', 'GAHE']
        # apply OCR
        terminated, truncated = False, False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            terminated = True
        
        return terminated, truncated, done_cap
    
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
            
        return True
        
    
def main():
    env = WebGame()
     
    # check if the environment is ok
    #env_checker.check_env(env)
    
    callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
    
    # Create the DQN model
    model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=520000,
                learning_starts=1000)
    
    # kick off training            
    model.learn(total_timesteps=200000, callback=callback, progress_bar=True)
       
    """ 
    # play 5 games
    for episode in range(5):
        ob  s = env.reset()
        done  = False
        total _reward = 0
        
        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            total_reward += reward
        print(f'Total reward for episode {episode} is {total_reward}')
    """
    
    
if __name__ == "__main__":
    main()