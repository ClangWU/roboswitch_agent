#@markdown ### **Imports**
# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from algorithm.dp.noisenet import ConditionalUnet1D
# env import
import algorithm.dp.dataset as ds
from huggingface_hub.utils import IGNORE_GIT_FOLDER_PATTERNS
import numpy as np
import matplotlib.pyplot as plt
import time
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
    is_cpu = False
    load_pretrained = True
    # parameters
    # 预测步长
    pred_horizon = 16
    # 观测步长 
    obs_horizon = 2
    # 动作步长
    action_horizon = 8
    obs_dim, action_dim = 10, 7
    batch_size = 256
    n_epochs = 100
    alpha = 0.001

    context_rep = zmq.Context()
    rep = context_rep.socket(zmq.REP)
    rep.bind("tcp://*:5555")  # 绑定端口5555

    context_req = zmq.Context()
    print("Connecting to server...")
    req = context_req.socket(zmq.REQ)
    req.connect("tcp://192.168.1.101:5555")  # 连接到服务器

    action_init = np.array([-0.00112, 0.00223919, -0.0140587, -0.00446902, 0.999931, -0.00453232, -0.00958188])

    scores_list = []
    actions_path = "./data/realrobot/actions.csv"
    states_path = "./data/realrobot/observations.csv"
    episode_ends_path = "./data/realrobot/episode_ends.csv"
    dataset = ds.FrankaDataset(
        actions_path, states_path, episode_ends_path,
        pred_horizon, obs_horizon, action_horizon)
    # save training data statistics (min, max) for each dim 
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True)
    
    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['obs'].shape:", batch['obs'].shape)
    print("batch['action'].shape", batch['action'].shape)

    # 网络接收动作维度作为输入，并将观测维度乘以观测时间范围作为全局条件
    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # example inputs
    # 生成带噪声的动作
    noised_action = torch.randn((1, pred_horizon, action_dim))
    # 生成观测
    obs = torch.zeros((1, obs_horizon, obs_dim))
    diffusion_iter = torch.zeros((1,))

    # the noise prediction network
    # takes noisy action, diffusion iteration and observation as input
    # predicts the noise added to action
    # 网络对带噪声的动作进行预测，生成噪声
    noise = noise_pred_net(
        sample=noised_action,
        timestep=diffusion_iter,
        global_cond=obs.flatten(start_dim=1))
    

    # illustration of removing noise
    # the actual noise removal is performed by NoiseScheduler
    # and is dependent on the diffusion noise schedule
    # 生成去噪声动作
    denoised_action = noised_action - noise

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    # 推理迭代次数
    num_diffusion_iters = 50
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # device transfer
    # move network to GPU
    device = torch.device('cuda')
    _ = noise_pred_net.to(device)

    if load_pretrained:
        ckpt_path = "./tmp/realrobot/cutting1_noise_pred_net.pt"
        state_dict = torch.load(ckpt_path, map_location='cuda')
        ema_noise_pred_net = noise_pred_net
        ema_noise_pred_net.load_state_dict(state_dict)
        print('Pretrained weights loaded.')
    else:
        print("Skipped pretrained weight loading.")

    #@markdown ### **Inference**
    # limit enviornment interaction to 200 steps before termination
    max_steps = 2000
    # use a seed >200 to avoid initial states seen in the training dataset
    
    # get first observation
    zmq_action(rep, action_init)
    time.sleep(0.1)
    a= time.time()

    observation = zmq_obs(req)

    print(time.time()-a)
    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [observation] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    rewards = list()
    done = False
    step_idx = 0
    score = 0
    with tqdm(total=max_steps, desc="cutting") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque)
            # normalize observation
            nobs = ds.normalize_data(obs_seq, stats=stats['obs'])
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            # time1 = time.time()
            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            # 将归一化的动作转换为原始动作
            action_pred = ds.unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            # 看着论文就知道下面的顺序是怎么来的了
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # 执行8个动作，不需要重新规划
            # without replanning
            # time2 = time.time()
            # delta = time2 - time1
        #   print("delta",delta)
            
            for i in range(len(action)):
                # print('action', action[i], 'step', step_idx, 'i', i)
                # stepping env
                zmq_action(rep, action[i])
                observation = zmq_obs(req)
                time.sleep(0.1)

                if observation[2] < -0.11:
                    done = True
                    break

                # save observations
                obs_deque.append(observation)
                # update progress bar
                step_idx += 1
                pbar.update(1)
                if step_idx > max_steps:
                    done = True
                if done:
                    break
                        

