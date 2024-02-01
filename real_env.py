import time
import numpy as np
import collections
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from frankx import Affine, JointMotion, LinearMotion, Robot

class RealEnv:
    """
    Environment for real robot
    Action space:       panda_joint (7)                # absolute joint position
                      
    Observation space: {"qpos": panda_joint (7),   # absolute joint position  (rad)
                        "wrench":  force(3)         # absolute joint velocity (rad)}
    """

    def __init__(self, dyn_rel=0.15, setup_robots=True):
        self.robot = None
        if setup_robots:
            self.setup_robots()
            self.robot.set_default_behavior()
            self.robot.recover_from_errors()
            self.robot.set_dynamic_rel(dyn_rel)

    def setup_robots(self):
        parser = ArgumentParser()
        parser.add_argument('--host', default='192.168.1.100', help='FCI IP of the robot')
        args = parser.parse_args()

        self.robot = Robot(args.host)

    def get_qpos(self):
        state = self.robot.read_once()
        return state.q

    def get_qvel(self):
        state = self.robot.read_once()
        return state.q_d
    
    def get_force(self):
        state = self.robot.read_once()
        
        return 

    def get_images(self):
        return self.image_recorder.get_images()

    def set_gripper_pose(self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized):
        left_gripper_desired_joint = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_desired_pos_normalized)
        self.gripper_command.cmd = left_gripper_desired_joint
        self.puppet_bot_left.gripper.core.pub_single.publish(self.gripper_command)

        right_gripper_desired_joint = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(right_gripper_desired_pos_normalized)
        self.gripper_command.cmd = right_gripper_desired_joint
        self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)

    def _reset_joints(self):
        reset_position = START_ARM_POSE[:6]
        move_arms([self.puppet_bot_left, self.puppet_bot_right], [reset_position, reset_position], move_time=1)

    def _reset_gripper(self):
        """Set to position mode and do position resets: first open then close. Then change back to PWM mode"""
        move_grippers([self.puppet_bot_left, self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)
        move_grippers([self.puppet_bot_left, self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=1)

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            # Reboot puppet robot gripper motors
            self.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
            self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
            self._reset_joints()
            self._reset_gripper()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def step(self, action):
        state_len = int(len(action) / 2)
        left_action = action[:state_len]
        right_action = action[state_len:]
        self.puppet_bot_left.arm.set_joint_positions(left_action[:6], blocking=False)
        self.puppet_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)
        self.set_gripper_pose(left_action[-1], right_action[-1])
        time.sleep(DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14) # 6 joint + 1 gripper, for two arms
    # Arm actions
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # Gripper actions
    action[6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
    action[7+6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

    return action


def make_real_env(init_node, setup_robots=True):
    env = RealEnv(init_node, setup_robots)
    return env


def test_real_teleop():
    """
    Test bimanual teleoperation and show image observations onscreen.
    It first reads joint poses from both master arms.
    Then use it as actions to step the environment.
    The environment returns full observations including images.

    An alternative approach is to have separate scripts for teleoperation and observation recording.
    This script will result in higher fidelity (obs, action) pairs
    """

    onscreen_render = True
    render_cam = 'cam_left_wrist'

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    setup_master_bot(master_bot_left)
    setup_master_bot(master_bot_right)

    # setup the environment
    env = make_real_env(init_node=False)
    ts = env.reset(fake=True)
    episode = [ts]
    # setup visualization
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam])
        plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam])
            plt.pause(DT)
        else:
            time.sleep(DT)


if __name__ == '__main__':
    test_real_teleop()