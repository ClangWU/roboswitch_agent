""" Test the behavioral cloning model on the PandaReach-v3 environment. """""
import torch as T
from algorithm.bc.behaviorcloning import BC_Agent
from joystick import FrankaPandaJoystickActor
import gymnasium as gym
import panda_gym
import matplotlib.pyplot as plt
if __name__ == '__main__':
    env = gym.make('PandaReach-v3', render_mode="human")
    N = 20
    batch_size = 256
    n_epochs = 10
    alpha = 0.003
    scores_list = []
    print(env.observation_space["observation"].shape)
    agent = BC_Agent(n_actions=env.action_space.shape[0], 
                    input_dims=env.observation_space["observation"].shape, batch_size=batch_size,
                    n_epochs=n_epochs)
    # Create the joystick actor
    # actor = FrankaPandaJoystickActor(env)
    # Train the behavioral cloning model
    model = agent.load_models()

    max_iters = 5000
    n_games = 20
    for i in range(n_games):
        observation, info = env.reset()
        env.render()  # Render the environment (optional)
        score = 0
        terminated= False
        truncated = False
        iters = 0
        while not terminated or not truncated:
            combined_observation = observation["observation"][0:3].tolist() + observation["desired_goal"][0:3].tolist()
            obs_tensor = T.tensor(combined_observation, dtype=T.float32).to(agent.bc.device)
            action = agent.take_action(obs_tensor*1000)
            observation, reward, terminated, truncated, info = env.step(action/1000)
            score += reward
            iters += 1
            if iters>max_iters:
                terminated= True
                truncated = True
        scores_list.append(score)
        print('Game:', i+1, 'Score:', score)

    env.close()
    games = list(range(1, n_games + 1))

    # 绘制得分曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(games, scores_list, marker='o', color='b')
    plt.title('Scores over Games')
    plt.xlabel('Game Number')
    plt.ylabel('Score')
    plt.grid(True)
    plt.xticks(games)
    plt.savefig('./plots/bc/bc.png')
    plt.show()
