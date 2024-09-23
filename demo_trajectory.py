import h5py
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from 
# def extract_trajectory(trajectory_data):
#     trajectories = []
#     for traj in trajectory_data:
#         np_obs = traj['obs']
#         np_obs_dict = np_obs.tolist()
#         # print(type(np_obs_dict))
#         obs = np_obs_dict['obs']
#         # print(f"The shape of obs_list is: {np.array(obs).shape}")
#         trajectories.append(obs)
#     print(f"The shape of obs_list is: {np.array(trajectories).shape}")
#     # print(np_obs_dict[0])
#     return trajectories
# pos_agent
def extract_trajectory(trajectory_data):
    trajectories = []
    trajectories_block = []
    pos_agents = []
    pos_blocks = []
    for traj in trajectory_data:
        np_info = traj['info']
        np_info_list = np_info.tolist()
        # print(np_info_list)
        # print(f"The shape of np_info_list is: {np.array(np_info_list).shape}")
        # import pdb; pdb.set_trace()
        # print(type(np_info_list))
        pos_agent = np_info_list[0]['pos_agent'][0]
        pos_agents.append(pos_agent)
        # for i, d in enumerate(np_info_list):
        #     pos_agent = d['pos_agent']
        #     # print((pos_agent))
        #     pos_agents.append(pos_agent)
        #     pos_block = d['block_pose']
        #     pos_blocks.append(pos_block)
    # print(f"The shape of obs_list is: {np.array(trajectories).shape}")
    # print(pos_agents)
    return pos_agents,trajectories_block

def convert_to_serializable(obj):
    """Convert non-serializable objects to serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Object not serializable")
def plot_2d_trajectory(ax, trajectories):
    """Plot multiple 2D trajectories from a 2d array."""
    # print(trajectories)
    # for i,traj_list in enumerate(trajectories): 
    #     traj_list = np.array(traj_list)
    #     ax.scatter(traj_list[0],traj_list[1], alpha=0.6)
    # ax.grid(True)  
      
    x_coords = [point[0] for point in trajectories]
    y_coords = [point[1] for point in trajectories]

# 绘制轨迹
    plt.plot(x_coords, y_coords, marker='o')
    
    # print('traj_list',traj_list)
    # num_trajectories = traj_list.shape[0]
    # for traj_index in range(num_trajectories):
    #     traj = traj_list[traj_index]
    #     print('traj',traj) 
    #     print(x1)
        # 绘制x1点 例如x1:308.66749160210713 397.1122982694017
        # ax.scatter(x2[0],x2[1], alpha=0.6)#绘制x2点
    
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.legend()
    

# def plot_3d_trajectory(ax, traj_list, label):
#     """Plot a 3D trajectory."""
#     # print("Shape of traj_list:", traj_list.shape) 
#     traj_list = np.array(traj_list)
#     num_frames = traj_list.shape[0]
#     for i in range(num_frames):
#         c = plt.cm.Blues(0.5 + 0.5 * i / num_frames)  # 使用蓝色渐变

#         marker = "o"  # 默认使用圆点标记

#         if i > 0:
#             v = np.array(traj_list[i]) - np.array(traj_list[i - 1])
#             ax.quiver(traj_list[i - 1][0], traj_list[i - 1][1], traj_list[i - 1][2],
#                       v[0], v[1], v[2], color="r", alpha=0.5)

#         ax.plot([traj_list[i][0]], [traj_list[i][1]], [traj_list[i][2]],
#                 marker=marker, label=label if i == 0 else "", color=c)

#     ax.legend()

# 读取轨迹数据
trajectory_data = []
trajectory_path = "data/pusht_eval_output/trajectory_data.hdf5"

with h5py.File(trajectory_path, 'r') as f:
    for idx in range(len(f)):
        traj = f[f"traj_{idx}"]
        obs = np.array(json.loads(traj['obs'][()]))
        action = np.array(traj['action'])
        reward = np.array(traj['reward'])
        done = np.array(traj['done'])
        info = np.array(json.loads(traj['info'][()]))
        trajectory_data.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'done': done,
            'info': info
        })

trajectories,_ = extract_trajectory(trajectory_data)
# print(np.array(trajectories).shape)
# import pdb; pdb.set_trace()
# 可视化轨迹
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# 2d 可视化
fig, ax = plt.subplots()
ax.set_xlim(0, 500)  # 根据数据调整范围
ax.set_ylim(0, 500)  # 根据数据调整范围
plot_2d_trajectory(ax, trajectories)
# for i,traj_data in enumerate(trajectories):  
#     # print(np.array(traj_data).shape)
#     plot_2d_trajectory(ax, traj_data)
    
plt.savefig("trajectories_plot_info.png")

    # plt.savefig(f"trajectory_{i+1}.png")
    # import pdb;  pdb.set_trace()
    # plt.close(fig) 
# plt.savefig("trajectories_plot_info.png")
    # plt.show()

# Save each trajectory as a separate image
# for i, traj_data in enumerate(trajectories):
#     fig, ax = plt.subplots()
#     plot_2d_trajectory(ax, traj_data)
#     plt.savefig(f"trajectory_{i+1}.png")
#     plt.close(fig)  # Close the figure to avoid displaying it