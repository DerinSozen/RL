import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self,env,alpha=0.2,gamma=0.9):
        self.env = env
        self.alpha = alpha
        self.gamma= gamma
        self.pos_space = np.linspace(-1.2, 0.6, 20)
        self.vel_space = np.linspace(-0.07, 0.07, 20)
        
        self.Q = np.zeros((len(self.pos_space),len(self.vel_space),3))
        
    def act(self,state,deterministic=False):
        pos,vel = state
        pos = np.digitize(pos,self.pos_space)
        vel = np.digitize(vel,self.vel_space)
        
        if deterministic:
            return np.argmax(self.Q[pos,vel,:])
        else:
            return self.env.action_space.sample()
    def learn(self,state,action,reward,new_state):
        pos,vel = state
        pos = np.digitize(pos,self.pos_space)
        vel = np.digitize(vel,self.vel_space)
        
        new_pos,new_vel = new_state
        new_pos = np.digitize(new_pos,self.pos_space)
        new_vel = np.digitize(new_vel,self.vel_space)
        
        self.Q[pos,vel,action] = (1-self.alpha)*self.Q[pos,vel,action] + self.alpha*(reward + self.gamma*np.max(self.Q[new_pos,new_vel,:]-self.Q[pos,vel,action]))
        
def main():
    env = gym.make('CliffWalking-v0')
    eval_env = gym.make('CliffWalking-v0')
    state = env.reset()[0]

    terminated = False
    episodes = 2000
    epsilon = 1
    epsilon_decay = 2/episodes
    a = 0.005
    discount_factor = 0.9
    actor = QLearning(env,alpha=a)
    
    eval_episodes = []
    for i in range(episodes):
        
        state = env.reset()[0]
        
        done = False
        truncated = False
        rewards = 0
        
        while(not done):
            if np.random.random() < epsilon:
                action = actor.act(state,deterministic=False)
            else:
                action = actor.act(state,deterministic=True)
            
            new_state, reward, done, truncated,_ = env.step(action)
            
            actor.learn(state,action,reward,new_state)
            
            state = new_state
            
            rewards += reward
            
            if done:
                print(f'Episode {i} finished after {rewards} timesteps')
            
        epsilon = max(0, epsilon - epsilon_decay)
        
        if i % 10 == 0:
            eval_reward = 0
            for j in range(5):
                eval_state = eval_env.reset()[0]
                done = False
                truncated = False
                while (not done and not truncated):
                    action = actor.act(eval_state,deterministic=True)
                    new_state, reward, done, truncated, _ = eval_env.step(action)
                    eval_state = new_state
                    eval_reward += reward
                
            eval_episodes.append((i,(eval_reward/5)))
        
    env.close()
    print(eval_episodes)
    pickle.dump(eval_episodes,open('q_learning_evals.pkl','wb'))
    
if __name__ == '__main__':
    main()
