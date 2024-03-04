""" Test the behavioral cloning model on the PandaReach-v3 environment. """""
import torch as T
from algorithm.bc.behaviorcloning import BC_Agent
from joystick import FrankaPandaJoystickActor
import matplotlib.pyplot as plt
import numpy as np
import zmq
import json
import time
def zmq_action(socket, action):
    message = socket.recv_string()
    action_list = action.tolist()
    # 将浮点数数组编码为JSON字符串
    message = json.dumps(action_list)
    socket.send_string(message)

def zmq_obs(socket):
    message = "get obs"
    socket.send_string(message)
    # 等待服务器响应
    reply = socket.recv_string()
    obs = json.loads(reply)
    return obs

if __name__ == '__main__':
    obs_dim, action_dim = (10,), 7
    batch_size = 256
    n_epochs = 10
    alpha = 0.003
    scores_list = []
    agent = BC_Agent(n_actions=action_dim, 
                    input_dims=obs_dim, batch_size=batch_size,
                    n_epochs=n_epochs, chkpt_dir='./tmp/realrobot')
    # Create the joystick actor
    # actor = FrankaPandaJoystickActor(env)
    # Train the behavioral cloning model
    model = agent.load_models()

    context_rep = zmq.Context()
    rep = context_rep.socket(zmq.REP)
    rep.bind("tcp://*:5555")  # 绑定端口5555

    context_req = zmq.Context()
    print("Connecting to server...")
    req = context_req.socket(zmq.REQ)
    req.connect("tcp://192.168.1.101:5555")  # 连接到服务器

    action_init = np.array([-0.00112, 0.00223919, -0.0140587, -0.00446902, 0.999931, -0.00453232, -0.00958188])

    zmq_action(rep, action_init)
    time.sleep(0.1)
    observation = zmq_obs(req)

    max_iters = 5000
    n_games = 20
    score = 0
    terminated= False
    truncated = False
    iters = 0
    while not terminated or not truncated:
        obs = T.tensor(observation, dtype=T.float32).to(agent.bc.device)
        action = agent.take_action(obs)
        # zmq_action(rep, action)
        observation = zmq_obs(req)
        time.sleep(0.1)

        if observation[2] < -0.11:
            terminated= True

        iters += 1
        if iters>max_iters:
            terminated= True
            truncated = True

    