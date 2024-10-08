import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import os
import h5py
import matplotlib.pyplot as plt
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from waypoint_extraction.extract_waypoints import optimize_waypoint_selection,dp_waypoint_selection, greedy_waypoint_selection, entropy_waypoint_selection,gripper_change_detect
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

class PushTKeypointsRunner_Replay(BaseLowdimRunner):
    def __init__(self,
            output_dir,
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
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )
        # print(env_fn.statelist)
        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        print("start replay")
        print("n_train",n_train)
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        print("n_test",n_test)
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.action_variance = []
        self.output_dir = output_dir
        self.const2x = False
        self.const4x = False
        self.var_speed = False
        self.avg_speed = 1
        self.scale = 0.3
        self.err_threshold = 5
        
    def pad_to_fixed_length(self,arr,arr2, target_length, fill_mode='zero'):
        """
        Pads the input array to the target length.
        
        Parameters:
        - arr: numpy array of shape (n, m) where n is the current length.
        - target_length: int, the desired length after padding.
        - fill_mode: str, 'zero' for zero padding, 'repeat' to repeat the last element.
        
        Returns:
        - padded_arr: numpy array of shape (target_length, m).
        """
        current_length = arr.shape[0]
        if current_length >= target_length:
            # Truncate if longer
            return arr[:target_length]
        else:
            # Padding
            padding_length = target_length - current_length
            if fill_mode == 'zero':
                # Zero padding
                padding = np.zeros((padding_length, arr.shape[1]))
            elif fill_mode == 'repeat':
                # Repeat the last element
                last_element = arr[-1].reshape(1, -1)
                padding = np.repeat(last_element, padding_length, axis=0)
            else:
                raise ValueError("Unsupported fill_mode. Use 'zero' or 'repeat'.")
            
            # Combine the original array with the padding
            padded_arr = np.vstack((arr, padding))
            padded_arr2 = np.vstack((arr2, padding))
        return padded_arr, padded_arr2  
    
    def waypoint(self,flatten_actions,flatten_variances):
        file_path_2 = os.path.join('data/pusht_eval_output-Original-speed', 'traj_points.npy')
        traj_points = np.load(file_path_2)
        flatten_actions = np.array(flatten_actions)
        flatten_variances = np.array(flatten_variances)
        # print(flatten_actions.shape)
        lens = 0
        total_actions_speed = 0
        total_actions = 0
        actions_ = []
        variances_ = []
        for traj_idx in range(flatten_actions.shape[0]):  
            # traj_actions = actions[:, traj_idx, :, :]  # 当前轨迹的动作
            # traj_variances = variances[:, traj_idx, :, :]
            # flat_actions = traj_actions.reshape(-1, 2)
            # flat_variances = traj_variances.reshape(-1, 2)
            flat_actions = flatten_actions[traj_idx]
            # print(len(flat_actions))
            flat_variances = flatten_variances[traj_idx]
            total_actions += len(flat_actions)
            avg_variances = np.mean(flat_variances, axis=-1)
            # # 最大最小归一化：
            # min_val = np.min(avg_variances)
            # max_val = np.max(avg_variances)
            # avg_variances_scaled = (avg_variances - min_val) / (max_val - min_val)
            # sigmoid：
            # avg_variances_scaled = 1 / (1 + np.exp(-avg_variances))
            # 使用 tanh 函数进行归一化
            # avg_variances_scaled = np.tanh(avg_variances)
            # # 标准化归一化
            mean_val = np.mean(avg_variances)
            std_val = np.std(avg_variances)
            avg_variances_scaled = (avg_variances - mean_val) / std_val
            # print(avg_variances.shape)
            waypoints, distance = optimize_waypoint_selection( # if it's too slow, use greedy_waypoint_selection
                    env=None,
                    actions=flat_actions,
                    gt_states=flat_actions,
                    err_threshold=self.err_threshold,
                    pos_only=True,
                    entropy=avg_variances_scaled,
                )
            waypoints = np.array(waypoints)
            print(f"2xspeed:{traj_points[traj_idx]/2},dis{traj_idx}:",(traj_points[traj_idx]/2)-len(waypoints))
            adjusted_flat_actions = flat_actions[waypoints,:]
            total_actions_speed+=len(waypoints)
            adjusted_flat_variances = flat_variances[waypoints,:]
            adjusted_flat_actions, adjusted_flat_variances = self.pad_to_fixed_length(arr = adjusted_flat_actions,arr2 = adjusted_flat_variances, target_length=304, fill_mode='repeat')
            adjusted_actions = adjusted_flat_actions.reshape(38, 8, 2)
            adjusted_variances = adjusted_flat_variances.reshape(38, 8, 2)
            actions_.append(adjusted_actions)
            variances_.append(adjusted_variances)
        actions_ = np.array(actions_).transpose(1, 0, 2, 3)
        variances_ = np.array(variances_).transpose(1, 0, 2, 3)
        # print(actions_.shape)
        avg_speed = total_actions/total_actions_speed
        return actions_, variances_,avg_speed
    
    def adaptive_sampling(self, target_speedup, flatten_actions, flatten_variances):
        target_speedup = target_speedup  # 平均x倍速
        # actions = np.array(actions)
        # variances = np.array(variances)
        print("adaptive_sampling speed",target_speedup)
        # 初始化存储调整后的轨迹
        adjusted_actions = []
        adjusted_variances = []
        flatten_actions = np.array(flatten_actions)
        flatten_variances = np.array(flatten_variances)
        # 遍历每条轨迹
        for traj_idx in range(flatten_actions.shape[0]):  
            # traj_actions = actions[:, traj_idx, :, :]  # 当前轨迹的动作
            # traj_variances = variances[:, traj_idx, :, :]  # 当前轨迹的方差

            # # 将轨迹从 (38, 8, 2) 展开成 (38*8, 2)
            # flat_actions = traj_actions.reshape(-1, 2)
            # flat_variances = traj_variances.reshape(-1, 2)
            # # print(np.min(np.abs(np.diff(flat_actions))))
            # # print(len(flat_actions))
            # print(flat_actions)
            # import pdb;pdb.set_trace()
            # 计算每个step的平均方差，shape: (38*8,)
            flat_actions = flatten_actions[traj_idx]
            flat_variances = flatten_variances[traj_idx]
            avg_variances = np.mean(flat_variances, axis=-1)
            min_variance = avg_variances.min() 
            max_variance = avg_variances.max()

            # 将方差映射到速度权重
            speed_weights = 1 + (max_variance - avg_variances) / (max_variance - min_variance + 1e-5)

            # 目标采样后的轨迹长度，缩短为目标速度倍数
            target_length = int(len(flat_actions) / target_speedup)
            
            # 计算累积权重并归一化到 [0, 1] 范围
            cumulative_weights = np.cumsum(speed_weights)
            cumulative_weights /= cumulative_weights[-1]

            # 在归一化后的累积权重上均匀采样 target_length 个点
            new_indices = np.searchsorted(cumulative_weights, np.linspace(0, 1, target_length))
            print(np.max(np.diff(new_indices)))

            # 根据采样索引选择相应的动作和方差
            adjusted_flat_actions = flat_actions[new_indices, :]
            adjusted_flat_variances = flat_variances[new_indices, :]
            # print(adjusted_flat_actions.shape)
            adjusted_flat_actions, adjusted_flat_variances = self.pad_to_fixed_length(arr = adjusted_flat_actions,arr2 = adjusted_flat_variances, target_length=int(304/target_speedup), fill_mode='repeat')
            # 将调整后的扁平轨迹恢复到 (38, new_steps, 2)
            new_steps = int(8 / target_speedup)  # 例如2倍速时，new_steps = 4
            adjusted_traj_actions = adjusted_flat_actions.reshape(38, new_steps, 2)
            adjusted_traj_variances = adjusted_flat_variances.reshape(38, new_steps, 2)
            # print(adjusted_traj_actions.shape)
            
            # 存储调整后的轨迹
            adjusted_actions.append(adjusted_traj_actions)
            adjusted_variances.append(adjusted_traj_variances)
            # print(len(adjusted_actions))

        # 合并所有调整后的轨迹
        adjusted_actions = np.array(adjusted_actions).transpose(1, 0, 2, 3)
        adjusted_variances = np.array(adjusted_variances).transpose(1, 0, 2, 3)
        # print(adjusted_actions.shape)
        # adjusted_actions = np.stack(adjusted_actions, axis=1)  # shape: (38, 56, new_steps, 2)
        # adjusted_variances = np.stack(adjusted_variances, axis=1)
        
        return adjusted_actions, adjusted_variances
    
    def crop(self,actions,variances,target_length):
        return actions[:target_length],variances[:target_length]
    
    def adjust_actions(self,actions,variances):
        file_path_2 = os.path.join('data/pusht_eval_output-Original-speed', 'traj_points.npy')
        traj_points = np.load(file_path_2)
        actions = np.array(actions)
        variances = np.array(variances)
        flatten_acts = []
        flatten_vars = []
        for traj_idx in range(actions.shape[1]):  
            traj_actions = actions[:, traj_idx, :, :]  # 当前轨迹的动作
            traj_variances = variances[:, traj_idx, :, :]  # 当前轨迹的方差

            # 将轨迹从 (38, 8, 2) 展开成 (38*8, 2)
            flat_actions = traj_actions.reshape(-1, 2)
            flat_variances = traj_variances.reshape(-1, 2)
            # print(traj_points[traj_idx])
            flat_actions,flat_variances = self.crop(actions=flat_actions,variances=flat_variances,target_length=traj_points[traj_idx])
            flatten_acts.append(flat_actions)
            flatten_vars.append(flat_variances)
        if self.err_threshold>0:
            actions_,variances_,speed_ = self.waypoint(flatten_actions=flatten_acts,flatten_variances=flatten_vars) 
            # print("speed",speed)
            # 将speed存入
        elif self.avg_speed>1:
            actions_,variances_ = self.adaptive_sampling(flatten_actions=flatten_acts, flatten_variances=flatten_vars,target_speedup=self.avg_speed)
            speed_ = self.avg_speed
        return actions_,variances_,speed_
    
    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        # print("n_inits",n_inits)
        n_chunks = math.ceil(n_inits / n_envs)
        # print("n_chunk",n_chunks)
        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        ## dataload
        data_replay = "data/pusht_replay/pusht_diffusion_noise_Var"

        # 指定文件保存路径
        file_path = os.path.join(data_replay, 'rollout_data.hdf5')
        print("_________________start_replay____________________")
        # 保存数据到指定路径
        with h5py.File(file_path, 'r') as f:
            step_keys = list(f['actions'].keys())
    
            # 动态确定 step_num
            step_num = len(step_keys)
            actions = [f['actions'][f"step_{i}"][:] for i in range(step_num)]
            variances = [f['variances'][f"step_{i}"][:] for i in range(step_num)]
            rewards = [f['rewards'][f"step_{i}"][:] for i in range(step_num)]
            observations = [f['observations'][f"step_{i}"][:] for i in range(step_num)]
    
            # 提取 info 中的内容
            # pos_agent = [f['info/pos_agent'][f"step_{i}"][:] for i in range(step_num)]
            # vel_agent = [f['info/vel_agent'][f"step_{i}"][:] for i in range(step_num)]
            # block_pose = [f['info/block_pose'][f"step_{i}"][:] for i in range(step_num)]
            # goal_pose = [f['info/goal_pose'][f"step_{i}"][:] for i in range(step_num)]
            # n_contacts = [f['info/n_contacts'][f"step_{i}"][:] for i in range(step_num)]
        
        # file_path = os.path.join('data/pusht_eval_output-Original-speed', 'learn_num.npy')
        # file_path_2 = os.path.join('data/pusht_eval_output-Original-speed', 'traj_points.npy')
        # learn_num = np.load(file_path)
        # traj_points = np.load(file_path_2)
        # print(learn_num)
        # print(traj_points)
        # print("learn_num.shape",learn_num.shape)
        # print("traj_points.shape",learn_num.shape)
        ## 采用新的avgspeed方法
        # if self.avg_speed>1:
        #     actions,variances = self.adaptive_sampling(target_speedup = self.avg_speed,actions = actions,variances = variances)
        
        # actions,variances,speed = self.waypoint(actions = actions,variances = variances)    
        # print(speed)
        # import pdb;pdb.set_trace()
        actions,variances,speed_ = self.adjust_actions(actions,variances)
        step_i = 0    
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            # print("this_global_slice",this_global_slice)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            # policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            sampled_action = []
            is_noise_step = 0
            step_num = 0
            i_num = 0
            # learn_num = np.zeros(56, dtype=int)

            while not done:
                Do = obs.shape[-1] // 2
                if step_i >= len(actions):
                    break
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                action = actions[step_i]
                action_variance = variances[step_i]
                if self.const2x==True:
                    action_downsampled = action[:, ::2, :]
                    variance_downsampled = action_variance[:, ::2, :]
                    zeros_vector = np.zeros((56, 4, 1))
                    action_and_noise = np.concatenate((action_downsampled, variance_downsampled,zeros_vector), axis=-1)
                elif self.const4x==True:
                    action_downsampled = action[:, ::4, :]
                    variance_downsampled = action_variance[:, ::4, :]
                    zeros_vector = np.zeros((56, 2, 1))
                    action_and_noise = np.concatenate((action_downsampled, variance_downsampled,zeros_vector), axis=-1)
                elif self.var_speed==True:
                    mean_variances = np.mean(variances, axis=(0, 2, 3))* self.scale##改
                    mean_variances_expanded = mean_variances.reshape(56, 1, 1)
                    mean_variances_expanded = np.repeat(mean_variances_expanded, 8, axis=1)  # 重复 8 次
                    action_and_noise = np.concatenate((action, action_variance,mean_variances_expanded), axis=-1)
                else:
                    zeros_vector = np.zeros((56,int(8/self.avg_speed), 1))
                    action_and_noise = np.concatenate((action, action_variance,zeros_vector), axis=-1)
                step_i = step_i+1    

                obs, reward, done, info = env.step(action_and_noise)
                # learn_num += (done == False)
                # learn_num -= (done==False)
                # done = (learn_num == 0)
                
                done = np.all(done)
                past_action = action
                step_num = step_num+1
                pbar.update(action.shape[1])
            pbar.close()
            os.makedirs(self.output_dir, exist_ok=True)
            
            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        total_traj = 0
        traj_lens = []
        for i, traj in enumerate(env.statelist):
            # Extract the points in the trajectory
            trajectory_points = [point[3]['pos_agent'] for point in traj[0]]
            trajectory_points = np.array(trajectory_points)
            traj_lens.append(len(trajectory_points))
            total_traj += len(trajectory_points)
            # Plot the trajectory
            plt.figure()  # Create a new figure for each trajectory
            plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], marker='o', markersize=3)
            
            # Set the title and labels
            plt.title(f'Trajectory {i+1}')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.grid(True)
            
            directory = 'data/pusht_replay/plot'
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Your plotting code here

            # Save the plot in the specified folder
            plt.savefig(f'{directory}/trajectory_{i+1}.png')
            
            # Optionally, display the plot
            # plt.show()
            
            # Close the plot to avoid overlapping of plots in the next iteration
            plt.close()
        #     
        # learn_num = learn_num*8
        # file_path = os.path.join(self.output_dir, 'learn_num.npy')
        # np.save(file_path, learn_num)
        print(traj_lens)
        file_path_2 = os.path.join(self.output_dir, 'traj_points.npy')
        np.save(file_path_2, traj_lens)
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        # print("n_inits",n_inits)
        log_data['avg_speed:']=speed_
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
            

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        log_data['total_step_of_56_traj'] = int(total_traj)
        # log_data['total_action_of_56_traj'] = int(traj_lens)
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data