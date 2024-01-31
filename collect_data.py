""" Collects expert demonstrations using a PID controller."""
import pandas as pd
import gym
from joystick import FrankaPandaJoystickActor, PIDController 

def collect_bc_demonstrations(env, actor, num_episodes, observations_csv, actions_csv):
    """ Collects expert demonstrations using a PID controller."""
    observations_data = {'observation': []}
    actions_data = {'action': []}
    user_input = input("Press 'y' to append data to existing files or any other key to create new files: ")
    if user_input.lower() == 'y':
        mode = 'a'
        flag=False
    else:
        mode = 'w'
        flag = True
    kp = 0.05  # Proportional gain
    ki = 0.001  # Integral gain
    kd = 0.3  # Derivative gain

    max_iters = 10000

    for episode in range(num_episodes):
        print("Episode number: ", episode)
        observation, info = env.reset()
        score = 0
        terminated = False
        truncated = False
        iters = 0
        print("Initial", observation["observation"][0:3])
        print("desired", observation["desired_goal"][0:3])

        pid_controller = PIDController(kp, ki, kd)

        while not terminated:
            current_position = observation["observation"][0:3]
            desired_position = observation["desired_goal"][0:3]

            error = desired_position - current_position

            # Compute control action using the PID controller
            action = pid_controller.compute_action(error)
            combined_observation = observation["observation"][0:3].tolist() + observation["desired_goal"][0:3].tolist()
            observations_data['observation'].append(combined_observation)
            actions_data['action'].append(action.tolist())
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            iters += 1
            if iters > max_iters:
                terminated = True
                truncated = True
        print(f"score: {score}")

    env.close()
    observations_df = pd.DataFrame(observations_data)
    actions_df = pd.DataFrame(actions_data)

    with open(observations_csv, 'w') as f:
        f.write('observation\n')
    with open(actions_csv, 'w') as f:
        f.write('action\n')

    observations_df.to_csv(observations_csv, index=False, mode=mode, header=flag)
    actions_df.to_csv(actions_csv, index=False, mode=mode, header=flag)

    print(f"Observations data saved to {observations_csv}")
    print(f"Actions data saved to {actions_csv}")

def collect_dp_demonstrations(env, actor, num_episodes, observations_csv, actions_csv, episode_ends_csv):
    """ Collects expert demonstrations using a PID controller."""
    observations_data = {'observation': []}
    actions_data = {'action': []}
    episode_ends = {'episode_ends': []}
    user_input = input("Press 'y' to append data to existing files or any other key to create new files: ")
    if user_input.lower() == 'y':
        mode = 'a'
        flag=False
    else:
        mode = 'w'
        flag = True
    kp = 0.05  # Proportional gain
    ki = 0.001  # Integral gain
    kd = 0.3  # Derivative gain

    max_iters = 10000
    counter = 0
    
    for episode in range(num_episodes):
        print("Episode number: ", episode)
        observation, info = env.reset()
        score = 0
        terminated = False
        truncated = False
        iters = 0
        print("Initial", observation["observation"][0:3])
        print("desired", observation["desired_goal"][0:3])

        pid_controller = PIDController(kp, ki, kd)

        while not terminated:
            counter += 1
            current_position = observation["observation"][0:3]
            desired_position = observation["desired_goal"][0:3]
            error = desired_position - current_position

            # Compute control action using the PID controller
            action = pid_controller.compute_action(error)
            combined_observation = observation["observation"][0:3].tolist() + observation["desired_goal"][0:3].tolist()
            _obs = combined_observation*1000
            observations_data['observation'].append(_obs)
            observation, reward, terminated, truncated, info = env.step(action)
            _act = action*1000
            actions_data['action'].append(_act.tolist())
            score += reward
            iters += 1
            if terminated == True:
                episode_ends['episode_ends'].append(counter)
            if iters > max_iters:
                terminated = True
                truncated = True
        print(f"score: {score}")

    env.close()

    observations_df = pd.DataFrame(observations_data)
    actions_df = pd.DataFrame(actions_data)
    episode_ends_df = pd.DataFrame(episode_ends)

    with open(observations_csv, 'w') as f:
        f.write('observation\n')
    with open(actions_csv, 'w') as f:
        f.write('action\n')
    with open(episode_ends_csv, 'w') as f:
        f.write('episode_ends\n')

    observations_df.to_csv(observations_csv, index=False, mode=mode, header=flag)
    actions_df.to_csv(actions_csv, index=False, mode=mode, header=flag)
    episode_ends_df.to_csv(episode_ends_csv, index=False, mode=mode, header=flag)

    print(f"Observations data saved to {observations_csv}")
    print(f"Actions data saved to {actions_csv}")
    print(f"Episode ends data saved to {episode_ends_csv}")
    
if __name__ == '__main__':
    import gymnasium as gym
    import panda_gym

    env = gym.make('PandaReach-v3', render_mode="human")

    # Create the joystick actor
    # actor = FrankaPandaJoystickActor(env)
    actor = None
    algo = 'dp'
    # Define the number of expert episodes and the CSV filename
    num_episodes = 100
    if algo is 'bc':
        collect_bc_demonstrations(env, actor, num_episodes,        
            './data/bc_observations.csv', 
            './data/bc_actions.csv')
    elif algo is 'dp':
        collect_dp_demonstrations(env, actor, num_episodes,        
            './data/dp_observations.csv', 
            './data/dp_actions.csv',
            './data/dp_episode_ends.csv')
