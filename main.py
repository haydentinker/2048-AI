from game import *
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import tensorflow as tf
import math
import torch
from pynput.keyboard import Key, Controller
import time 
from PIL import Image
import torchvision.transforms as transforms
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
class GameEnv(Env):
    def __init__(self):
        # Actions we can move left, right, up, and down
        # pygame.init()
        # placerandomtile()
        # placerandomtile()
        # self.move(1)
        time.sleep(.5)
        transform = transforms.Compose([transforms.PILToTensor()])
        self.action_space = Discrete(4)
        pygame.image.save(surface, "state.jpg")
        self.image=Image.open('state.jpg')
        self.state=transform(self.image)
        print(self.state.size())
        board_size=4
        self.normalGrid=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        #self.observation_space = Box(low=0,high=1,shape=(4,4))
        self.observation_space = Box(low=0, high=255, shape=(self.state.size()), dtype=np.uint8)
        self.loseStatus=False
        self.previousTotal=0
        self.reward=0
        # Get the RGB values for the screen shot
        # Starting state could be a picture of the environment
       
    def step(self, action):
        # Apply action
        # 0 = move left
        # 1 = move right
        # 2 = move up
        # 3 = move down
        #time.sleep(.25)
        if self.first:
            pygame.init()
            placerandomtile()
            placerandomtile()
            self.first=0
        printmatrix()
        self.move(action)
        # self.normalizeGrid()
        # normalGrid=self.normalGrid
        # flatGrid = sum(normalGrid, [])
        transform = transforms.Compose([transforms.PILToTensor()])
        pygame.image.save(surface, "state.jpg")
        self.image=Image.open('state.jpg')
        self.state=transform(self.image)
        # Reward if score is greater than last time
        total=getTotalPoints()
        reward=total-self.previousTotal
        self.previousTotal=total
        # Check if game over
        done=self.loseStatus
        if done:
            gameover()
        info = {}
        
        # Return step information
        return self.state,reward, done, info
    def normalizeGrid(self):
        tileofmatrix=getTileMatrix()
        for i in range(4):
            for j in range(4):
                if not tileofmatrix[i][j]==0:
                    self.normalGrid[i][j]=(math.log2(int(tileofmatrix[i][j])))/17
    def normalizeReward(self):
        self.normalReward=math.log2(self.reward)
    def move(self,action):
        if action==0:
            kb.press(Key.up) # Presses "up" key
            kb.release(Key.up) # Releases "up" key
        elif action==1:
            kb.press(Key.down) # Presses "up" key
            kb.release(Key.down) # Releases "up" key
        elif action==2:
            kb.press(Key.left) # Presses "up" key
            kb.release(Key.left) # Releases "up" key
        elif action==3:
            kb.press(Key.right) # Presses "up" key
            kb.release(Key.right) # Releases "up" key
        for event in pygame.event.get():
            if checkIfCanGo() == True:
                if event.type == KEYDOWN:
                    if isArrow(event.key):
                        rotations = getrotations(event.key)
                        addToUndo()
                        for i in range(0,rotations):
                            rotatematrixclockwise()

                        if canmove():
                            movetiles()
                            self.reward=mergetiles()
                            placerandomtile()

                        for j in range(0,(4-rotations)%4):
                            rotatematrixclockwise() 
                        printmatrix()
            else:
                self.loseStatus=True
        
        pygame.display.update()
    def render(self, mode="human", close=False):
        #time.sleep(.25)
        printmatrix()
        return 0
    
    def reset(self):
        reset()
        self.first=1
        transform = transforms.Compose([transforms.PILToTensor()])
        pygame.image.save(surface, "state.jpg")
        self.image=Image.open('state.jpg')
        self.state=transform(self.image)
        self.loseStatus=False
        return self.state
    
kb=Controller()
env=GameEnv()
env=DummyVecEnv([lambda:env])
model=PPO('MlpPolicy',env,verbose=1)
model.learn(total_timesteps=40000)
model.save("ppo-2048")
#model.load("ppo-2048")

evaluate_policy(model,env,n_eval_episodes=10,render=True)
obs=env.reset()
print("Running Game")
done=false
score=0
while not done:
    time.sleep(.25)
    env.render()
    action,_states=model.predict(obs)
    obs,rewards,done,info=env.step(action)
    score+=rewards
    
env.close()
# def build_agent(model, actions):
#     policy = BoltzmannQPolicy()
#     memory = SequentialMemory(limit=50000, window_length=1) #This stores the agent's experience
#     dqn = DQNAgent(model=model, memory=memory, policy=policy, 
#                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
#     return dqn
# states=env.observation_space.shape
# actions=env.action_space.n
# print(states)
# model = Sequential()
# model.add(Dense(24,activation='relu',input_shape=states))
# model.add(Dense(24,activation='relu'))
# model.add(Dense(24,activation='relu'))
# model.add(Dense(24,activation='relu'))
# model.add(Dense(actions,activation='linear'))
# print(tuple(model.output_shape))
# dqn = build_agent(model, actions)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
# scores = dqn.test(env, nb_episodes=100, visualize=False)
# print(np.mean(scores.history['episode_reward']))