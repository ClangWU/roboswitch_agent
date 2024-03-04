""" Train the behavioral cloning model """
import torch
from algorithm.bc.behaviorcloning import BC_Agent
from joystick import FrankaPandaJoystickActor

if __name__ == '__main__':
    obs_dim, action_dim = (10,), 7
    batch_size = 256
    n_epochs = 200
    alpha = 0.001
    # print(obs_shape)
    agent = BC_Agent(n_actions=action_dim, 
                    input_dims=obs_dim, batch_size=batch_size,
                    n_epochs=n_epochs, chkpt_dir='./tmp/realrobot')

    # # # Train the behavioral cloning model
    model = agent.train_behavioral_cloning(
        './data/realrobot/observations.csv', 
        './data/realrobot/actions.csv')

    agent.save_models()




