import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
@click.command()
@click.option('-c', '--datapath', required=True)
@click.option('-d', '--device', default='cuda:0')

## 这里的datapath就是你存储的路径
def main(datapath, device):
    keypoint_visible_rate=1.0,
    n_train=10,
    n_train_vis=3,
    train_start_seed=0,
    n_test=22,
    n_test_vis=6,
    legacy_test=False,
    test_start_seed=10000,
    max_steps=200,
    n_obs_steps=8,
    n_action_steps=8,
    n_latency_steps=0,
    fps=10,
    crf=22,
    agent_keypoints=False,
    past_action=False,
    tqdm_interval_sec=5.0,
    # n_envs=None
    
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
                    )
    # obs/state -> image img_list video
    
    end = replay(datapath,env)
    
def plot_traj(trajectory_points,rollout_id):
    i = rollout_id
    trajectory_points = np.array(trajectory_points)
    output_directory = 'pusht/plot/traj'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
            
    # Plot the trajectory
    plt.figure()  # Create a new figure for each trajectory
    plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], marker='o', markersize=3)
    
    # Set the title and labels
    plt.title(f'Trajectory {i+1}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(output_directory, f'trajectory_{i+1}.png'))
    
    # Optionally, display the plot
    # plt.show()
    
    # Close the plot to avoid overlapping of plots in the next iteration
    plt.close()
    return True

def plot_var(trajectory_points, var_s, rollout_id):
    i = rollout_id
    trajectory_points = np.array(trajectory_points)
    var_s = np.array(var_s)
    avg_vars = (var_s[:, 0] + var_s[:, 1]) / 2.0
    # Ensure the output directory exists
    output_directory = 'pusht/plot/var'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Create segments for color mapping
    points = trajectory_points.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Normalize variance values for color mapping
    norm = Normalize(vmin=np.min(avg_vars), vmax=np.max(avg_vars))
    
    # Create a line collection with the segments and corresponding colors
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(avg_vars)
    lc.set_linewidth(2)
    
    # Plot the trajectory with variance-based coloring
    plt.figure()
    plt.gca().add_collection(lc)
    plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], 'o', markersize=3, color='black')  # Add points as markers
    plt.colorbar(lc, label='Variance')  # Add a colorbar to show variance scale
    
    # Set the title and labels
    plt.title(f'Trajectory Variance {i+1}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    
    # Set axis limits based on the trajectory points
    plt.xlim(trajectory_points[:, 0].min(), trajectory_points[:, 0].max())
    plt.ylim(trajectory_points[:, 1].min(), trajectory_points[:, 1].max())
    
    # Save the figure
    plt.savefig(os.path.join(output_directory, f'trajectory_variance_{i+1}.png'))
    
    # Optionally, display the plot
    # plt.show()
    
    # Close the plot to avoid overlapping in the next iteration
    plt.close()
    return True


def replay(datapath,env):
    n_train = 6
    n_test = 50
    n_envs = None
    if n_envs is None:
        n_envs = n_train + n_test

    
    for traj_num in range(n_envs):
        # Open the HDF5 file for loading demos
        dataset_path = os.path.join(datapath, f"replay_{traj_num+1}.hdf5")
        with h5py.File(dataset_path, "r") as hdf5_file:
            obs_dataset = hdf5_file['obs']
            action_dataset = hdf5_file['actions']
            reward_dataset = hdf5_file['reward']
            pos_dataset = hdf5_file['info/pos_agent']
            vel_dataset = hdf5_file['info/vel_agent']
            block_pose_dataset = hdf5_file['info/block_pose']
            goal_pose_dataset = hdf5_file['info/goal_pose']
            n_contacts_dataset = hdf5_file['info/n_contacts']
            var_dataset = hdf5_file['vars']
            
            ######## 如果要画轨迹直接调用plot_traj
            x = plot_traj(pos_dataset,traj_num)
            
            ######## 如果要画带var的轨迹调用
            y=plot_var(pos_dataset, var_dataset, traj_num)
            # Running in open-loop using the saved demos
            env.reset()
            for step in range(len(obs_dataset)):
                obs = obs_dataset[step]
                action = action_dataset[step]
                reward = reward_dataset[step]
                pos = pos_dataset[step]
                var = var_dataset[step]

                # Set the environment to the saved observation
                # env.set_state(obs)  # You need to implement this method if not provided
                # Step the environment with the saved action
                next_obs, reward, done, info = env.step(action)

                # Optionally, compare or process the rewards, info, etc.
                # print(f"Step {step}: Obs: {obs}, Action: {action}, Reward: {reward}, Info: {info}")
                
                
                
                
    return 0
        
if __name__ == '__main__':
    main()