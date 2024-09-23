import os
import numpy as np
file_path_2 = os.path.join('data/pusht_eval_output-Original-speed', 'traj_points.npy')
traj_points = np.load(file_path_2)
print(traj_points)