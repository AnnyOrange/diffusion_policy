import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
data_replay = "data/pusht_replay/pusht_diffusion_noise_Var"

# 指定文件保存路径
file_path = os.path.join(data_replay, 'rollout_data.hdf5')
print("_________________distributions____________________")
# 保存数据到指定路径
with h5py.File(file_path, 'r') as f:
    step_keys = list(f['actions'].keys())

    # 动态确定 step_num
    step_num = len(step_keys)
    variances = [f['variances'][f"step_{i}"][:] for i in range(step_num)]
variances = np.array(variances)
num_trajectories = variances.shape[1]  # 56 trajectories

# Create individual histograms for each trajectory's variances
save_directory = 'distribution/vars_mean'
os.makedirs(save_directory, exist_ok=True)
for i in range(num_trajectories):
    # Extract variances for each trajectory, reshape to (38*8, 2) for x and y axes
    trajectory_variance = variances[:, i, :, :].reshape(-1, 2)
    x_variances = trajectory_variance[:, 0]
    y_variances = trajectory_variance[:, 1]
    mean_variances = (trajectory_variance[:, 0] + trajectory_variance[:, 1]) / 2
    
    # 将大于 100 的方差值截断为 100
    mean_variances_clipped = np.clip(mean_variances, 0, 1000)
    
    # 绘制 mean_variances 的分布图
    plt.figure(figsize=(10, 6))
    plt.hist(mean_variances_clipped, bins=30, alpha=0.6, label='Mean Variances (Clipped at 1000)', density=True, color='green')
    plt.title(f'Mean Variance Distribution for Trajectory {i + 1}')
    plt.xlabel('Mean Variance Value')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表为 PNG 文件
    save_path = os.path.join(save_directory, f'trajectory_mean_variance_{i + 1}.png')
    plt.savefig(save_path)
    plt.close()