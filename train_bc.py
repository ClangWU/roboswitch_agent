""" Train the behavioral cloning model """
import torch
from algorithm.bc.behaviorcloning import BC_Agent
from joystick import FrankaPandaJoystickActor
import gymnasium as gym
import panda_gym

def get_env_info():
    env = gym.make('PandaReach-v3', render_mode="human")
    obs_dim = env.observation_space["observation"].shape
    action_dim = env.action_space.shape[0]
    env.close()
    print(obs_dim, action_dim)
    return obs_dim, action_dim

if __name__ == '__main__':
    obs_dim, action_dim = get_env_info()
    batch_size = 32
    n_epochs = 100
    alpha = 0.001
    # print(obs_shape)
    agent = BC_Agent(n_actions=action_dim, 
                    input_dims=obs_dim, batch_size=batch_size,
                    n_epochs=n_epochs, chkpt_dir='./tmp/bc')

    # # # Train the behavioral cloning model
    model = agent.train_behavioral_cloning(
        './data/bc/bc_observations.csv', 
        './data/bc/bc_actions.csv')
    agent.save_models()

    agent.load_models()
    env = gym.make('PandaReach-v3', render_mode="human")
    max_iters = 5000
    n_games = 10
    for i in range(n_games):
        observation, info = env.reset()
        env.render()  # Render the environment (optional)
        score = 0
        terminated= False
        truncated = False
        iters = 0
        while not terminated or not truncated:
            combined_observation = observation["observation"][0:3].tolist() + observation["desired_goal"][0:3].tolist()
            obs_tensor = torch.tensor(combined_observation, dtype=torch.float32).to(agent.bc.device)
            action = agent.take_action(obs_tensor*1000)
            observation, reward, terminated, truncated, info = env.step(action/1000)
            score += reward
            iters += 1
            if iters>max_iters:
                terminated= True
                truncated = True
        print('Game:', i+1, 'Score:', score)

    env.close()


