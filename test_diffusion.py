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
import gymnasium as gym
import panda_gym
import matplotlib.pyplot as plt
import time
def get_env_info():
    env = gym.make('PandaReach-v3', render_mode="human")
    obs_dim = env.observation_space["observation"].shape
    action_dim = env.action_space.shape[0]
    env.close()
    print(obs_dim, action_dim)
    return obs_dim, action_dim

if __name__ == '__main__':
    is_cpu = False
    load_pretrained = True
    n_games = 10
    # parameters
    # 预测步长
    pred_horizon = 16
    # 观测步长 
    obs_horizon = 2
    # 动作步长
    action_horizon = 8
    obs_dim, action_dim = get_env_info()
    batch_size = 256
    n_epochs = 100
    alpha = 0.001
    scores_list = []
    actions_path = "./data/dp/dp_actions.csv"
    states_path = "./data/dp/dp_observations.csv"
    episode_ends_path = "./data/dp/dp_episode_ends.csv"
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

    # observation and action dimensions corrsponding to
    obs_dim = 6
    action_dim = 3

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
    if is_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    _ = noise_pred_net.to(device)
    num_epochs = 100

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    # 初始化一个 EMA 模型，用于跟踪网络权重的移动平均。这有助于稳定训练过程并加速收敛。
    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)

    if load_pretrained:
      ckpt_path = "./tmp/dp/ddim_noise_pred_net.pt"
    #   ckpt_path = "./tmp/dp/ddim_22_noise_pred_net.pt"
      # if not os.path.isfile(ckpt_path):
      #     id = "1mHDr_DEZSdiGo9yecL50BBQYzR8Fjhl_&confirm=t"
      #     gdown.download(id=id, output=ckpt_path, quiet=False)
      if is_cpu:
            state_dict = torch.load(ckpt_path, map_location='cpu') 
            print('Use CPU')
      else:
            state_dict = torch.load(ckpt_path, map_location='cuda')
            print('Use GPU')
      ema_noise_pred_net = noise_pred_net
      ema_noise_pred_net.load_state_dict(state_dict)
      print('Pretrained weights loaded.')
    else:
      print("Skipped pretrained weight loading.")

    #@markdown ### **Inference**
    # limit enviornment interaction to 200 steps before termination
    max_steps = 300
    env = gym.make('PandaReach-v3', render_mode="human")
    # use a seed >200 to avoid initial states seen in the training dataset
    
    for id in range(n_games):
      # get first observation
      observation, info = env.reset()
      obs = observation["observation"][0:3].tolist() + observation["desired_goal"][0:3].tolist()
      # keep a queue of last 2 steps of observations
      obs_deque = collections.deque(
          [obs] * obs_horizon, maxlen=obs_horizon)
      # save visualization and rewards
      rewards = list()
      done = False
      step_idx = 0
      score = 0
      with tqdm(total=max_steps, desc="Franka Reach") as pbar:
          while not done:
              B = 1
              # stack the last obs_horizon (2) number of observations
              obs_seq = np.stack(obs_deque)
              # normalize observation
              nobs = ds.normalize_data(obs_seq, stats=stats['obs'])
              # device transfer
              nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
              time1 = time.time()
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
              time2 = time.time()
              delta = time2 - time1
            #   print("delta",delta)
                
              for i in range(len(action)):
                  # print('action', action[i], 'step', step_idx, 'i', i)
                  # stepping env
                  
                  observation, reward, done, _, info = env.step(action[i])
                #   print("time2",time.time())
                  obs = observation["observation"][0:3].tolist() + observation["desired_goal"][0:3].tolist()

                  # save observations
                  obs_deque.append(obs)
                  # and reward/vis
                  score += reward
                  rewards.append(reward)
                  # update progress bar
                  step_idx += 1
                  pbar.update(1)
                #   pbar.set_postfix(reward=reward)
                  if step_idx > max_steps:
                      done = True
                  if done:
                      break
      scores_list.append(score)
      # print out the score
      print('Game:', id+1, 'Score:', score)

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
    plt.savefig('./plots/dp/dp.png')
    plt.show()

