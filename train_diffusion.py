#@markdown ### **Imports**
# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from algorithm.dp.noisenet import ConditionalUnet1D
# env import
import pygame
import algorithm.dp.dataset as ds
from huggingface_hub.utils import IGNORE_GIT_FOLDER_PATTERNS
import numpy as np
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
    actions_path = "./data/dp_actions.csv"
    states_path = "./data/dp_observations.csv"
    episode_ends_path = "./data/dp_episode_ends.csv"
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
    # 
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
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
    num_epochs = 100

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    # 初始化一个 EMA 模型，用于跟踪网络权重的移动平均。这有助于稳定训练过程并加速收敛。
    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    # Note that the scheduler is stepped every batch
    # 设置一个余弦学习率调度器，包含线性预热步骤。
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    train_model = True
    # training loop
    if train_model:
        with tqdm(range(num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # data normalized in dataset
                        # device transfer
                        # shape: 256(batch size) *  2 * 6
                        nobs = nbatch['obs'].to(device)
                        # shape: 256(batch size) * 16 * 3
                        naction = nbatch['action'].to(device)
                        B = nobs.shape[0]

                        # observation as FiLM conditioning
                        # (B, obs_horizon, obs_dim)
                        obs_cond = nobs[:,:obs_horizon,:]
                        # (B, obs_horizon * obs_dim)  合并在一起展平以适配网络输入。
                        obs_cond = obs_cond.flatten(start_dim=1)

                        # sample noise to add to actions  生成噪声
                        noise = torch.randn(naction.shape, device=device) 

                        # sample a diffusion iteration for each data point 
                        # 随机选择扩散时间步 (batch size) 
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps,
                            (B,), device=device
                        ).long()

                        # add noise to the clean images according to the noise magnitude at each diffusion iteration
                        # (this is the forward diffusion process)
                        noisy_actions = noise_scheduler.add_noise(
                            naction, noise, timesteps)

                        # predict the noise residual
                        noise_pred = noise_pred_net(
                            noisy_actions, timesteps, global_cond=obs_cond)

                        # L2 loss
                        loss = nn.functional.mse_loss(noise_pred, noise)

                        # optimize
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        lr_scheduler.step()

                        # update Exponential Moving Average of the model weights
                        ema.step(noise_pred_net.parameters())

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))
        ema_noise_pred_net = noise_pred_net
        ema.copy_to(ema_noise_pred_net.parameters())

        # save model
        torch.save(ema_noise_pred_net.state_dict(), "./tmp/dp/noise_pred_net.pt")