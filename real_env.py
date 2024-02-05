import time
import numpy as np
import collections
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import math
from frankx import Affine, JointMotion, LinearMotion, Robot
from constants import HANDLOAD, START_ARM_POSE, END_ARM_POSE

class RealEnv:
    """
    Environment for real robot
    Action space:       panda_joint (7)                # absolute joint position
                      
    Observation space: {"qpos": panda_joint (7),     # absolute joint position  (rad)
                        "wrench":  force(3)          # force  (N)
                        "cut_height": cut_height (1) # cut height (m)}
    """

    def __init__(self, dyn_rel=0.15, setup_robots=True):
        self.robot = None
        if setup_robots:
            self.setup_robots()
            self.robot.set_default_behavior()
            self.robot.recover_from_errors()
            self.robot.set_dynamic_rel(dyn_rel)
        self.time_sec = 0
        self.start_time = 0
        self.now_time = 0

    def setup_robots(self):
        parser = ArgumentParser()
        parser.add_argument('--host', default='192.168.1.100', help='FCI IP of the robot')
        args = parser.parse_args()
        self.robot = Robot(args.host)
        self.gripper = self.robot.get_gripper()

    def get_qpos(self):
        state = self.robot.read_once()
        return state.q

    def get_qvel(self):
        state = self.robot.read_once()
        return state.q_d
    
    def get_force(self):
        state = self.robot.read_once()
        _Gravity = np.array(HANDLOAD)
        full_matrix = np.array(state.O_T_EE).reshape(4, 4)  # 假设 O_T_EE 是一个长度为 16 的列表或数组
        rotation_matrix = full_matrix[0:3, 0:3]
        GinF = rotation_matrix.T @ _Gravity

        compensated_force = np.array([
        state.K_F_ext_hat_K[0] + GinF[0],
        state.K_F_ext_hat_K[1] + GinF[1],
        state.K_F_ext_hat_K[2] + GinF[2]
        ])
        force_in_world = rotation_matrix @ compensated_force
        return force_in_world
    
    def set_gripper(self, pos):
        pos = abs(pos)
        if pos > 0.085:
            pos = 0.085
        self.gripper.move(pos)

    def reset_joints(self):
        reset_position = START_ARM_POSE
        self.robot.move(JointMotion(reset_position))

    def reset_gripper(self):
        self.gripper.move(0.085)

    def get_observation(self, state):
        if state == None:
            state = self.robot.read_once()
        _Gravity = np.array(HANDLOAD)
        full_matrix = np.array(state.O_T_EE).reshape(4, 4)  
        rotation_matrix = full_matrix[0:3, 0:3]
        GinF = rotation_matrix.T @ _Gravity

        compensated_force = np.array([
        state.K_F_ext_hat_K[0] + GinF[0],
        state.K_F_ext_hat_K[1] + GinF[1],
        state.K_F_ext_hat_K[2] + GinF[2]
        ])
        force_in_world = rotation_matrix @ compensated_force

        obs = collections.OrderedDict()
        obs['qpos'] = state.q
        obs['force'] = force_in_world
        obs['cut_height'] = state.O_T_EE[14]
        return obs

    def get_reward(self, state):
        if state == None:
            state = self.robot.read_once()
        return 0

    def reset(self):
        self.start_time = time.time()
        self.reset_joints()
        observation=self.get_observation()
        info = {}
        return observation, info

    def is_terminal(self, state):
        status = True
        # 切水果切到最底部
        if state.O_T_EE[14] < 0.05:
            state = False
        return status 

    def is_truncated(self, state):
        x_force = state.K_F_ext_hat_K[0]
        y_force = state.K_F_ext_hat_K[1]
        z_force = state.K_F_ext_hat_K[2]
        force_magnitude = math.sqrt(x_force**2 + y_force**2 + z_force**2)
        
        self.now_time = time.time()
        self.time_sec = self.now_time - self.start_time
        # force > 20 or time > 120s
        if force_magnitude > 20 or self.time_sec > 120:
            return True
        return False

    def step(self, action):
        # execute action
        self.robot.move(JointMotion(action))
        # get info
        state = self.robot.read_once()
        observation = self.get_observation(state)
        reward=self.get_reward(state)
        terminated = self.is_terminal(state)
        truncated = self.is_truncated(state)
        info = {}
        return observation, reward, terminated, truncated, info

    def auto_finish(self):
        self.robot.move(LinearMotion(END_ARM_POSE))

    def close(self):
        print("Closing the environment")

# check time_sec
# check force_magnitude
# check cut_height
