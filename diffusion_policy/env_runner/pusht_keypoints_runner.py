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

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

class PushTKeypointsRunner(BaseLowdimRunner):
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
        # n_chunks = 1
        data_replay = "data/pusht_replay/pusht_diffusion2x"
        os.makedirs(data_replay, exist_ok=True)  # 如果目录不存在，则创建

        # 指定文件保存路径
        file_path = os.path.join(data_replay, 'rollout_data.hdf5')

        # 保存数据到指定路径
        with h5py.File(file_path, 'w') as f:
            # 创建存储数据的组
            action_group = f.create_group('actions')
            variance_group = f.create_group('variances')
            reward_group = f.create_group('rewards')
            obs_group = f.create_group('observations')
            # info_group = f.create_group('info')

            # pos_agent_group = info_group.create_group('pos_agent')
            # vel_agent_group = info_group.create_group('vel_agent')
            # block_pose_group = info_group.create_group('block_pose')
            # goal_pose_group = info_group.create_group('goal_pose')
            # n_contacts_group = info_group.create_group('n_contacts')
            
            for chunk_idx in range(n_chunks):
                
                start = chunk_idx * n_envs
                # print("start",start)
                end = min(n_inits, start + n_envs)
                # print("end",end)
                this_global_slice = slice(start, end)
                # print("this_global_slice",this_global_slice)
                this_n_active_envs = end - start
                this_local_slice = slice(0,this_n_active_envs)
                
                this_init_fns = self.env_init_fn_dills[this_global_slice]
                # print("this_init_fns",this_init_fns)
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
                policy.reset()

                pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}", 
                    leave=False, mininterval=self.tqdm_interval_sec)
                done = False
                sampled_action = []
                is_noise_step = 0
                step_num = 0
                i_num = 0
                while not done:
                    Do = obs.shape[-1] // 2
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

                    # run policy
                    # with torch.no_grad():
                    #     action_dict = policy.predict_action(obs_dict)
                    # # device_transfer
                    # np_action_dict = dict_apply(action_dict,
                    #     lambda x: x.detach().to('cpu').numpy())
                    # #to simulate latency
                    # action = np_action_dict['action'][:,self.n_latency_steps:]
                    
                    
                    
                    with torch.no_grad(): #(56,8,2) (20,56,8,2)
                        action_samples = []
                        var_noises = []
                        for _ in range(20):  # Sample 20 actions for variance calculation
                            action_dict = policy.predict_action(obs_dict)
                            np_action_dict = dict_apply(action_dict,lambda x: x.detach().to('cpu').numpy())
                            action = np_action_dict['action'][:,self.n_latency_steps:]
                            action_samples.append(action)
                            # var_noises.append(var_noise_)
                        action = action_samples[0]
                        # var_noise = var_noises[0]
                        # print("var_noise",var_noise)
                        # action_var = action形状的，内容全是var_noise的
                        action_variance = np.var(action_samples, axis=0) #(56,8,2)
                        mean_variance = np.mean(action_variance, axis=1, keepdims=True)
                        mean_variance = np.repeat(mean_variance, 8, axis=1)
                                    
                    # noise_step = np.zeros_like(action)  
                    # if (step_num == is_noise_step):
                    #     noise_step[:, i_num, :] = 1  
                    # print("noise_step",noise_step)  
                    # step env
                    action_and_noise = np.concatenate((action, action_variance), axis=-1)
                    # print("actionandvar.shape",action_and_var.shape)

                    obs, reward, done, info = env.step(action_and_noise)
                    
                    done = np.all(done)
                    
                    ## TODO save obs, reward, done, info，SAVE action,  action_variance and mean_variance
                    action_group.create_dataset(f"step_{step_num}", data=action)
                    variance_group.create_dataset(f"step_{step_num}", data=action_variance)
                    reward_group.create_dataset(f"step_{step_num}", data=reward)
                    obs_group.create_dataset(f"step_{step_num}", data=obs)

                    # # 存储 info 中的内容
                    # pos_agent = [point['pos_agent'] for point in info]
                    # pos_agent_group.create_dataset('data', data=np.array(pos_agent, dtype=object))

                    # vel_agent = [point['vel_agent'] for point in info]
                    # vel_agent_group.create_dataset('data', data=np.array(vel_agent, dtype=object))

                    # block_pose = [point['block_pose'] for point in info]
                    # block_pose_group.create_dataset('data', data=np.array(block_pose, dtype=object))

                    # goal_pose = [point['goal_pose'] for point in info]
                    # goal_pose_group.create_dataset('data', data=np.array(goal_pose, dtype=object))

                    # n_contacts = [point['n_contacts'] for point in info]
                    # n_contacts_group.create_dataset('data', data=np.array(n_contacts, dtype=object))
                    past_action = action
                    step_num = step_num+1
                    # update pbar
                    pbar.update(action.shape[1])
                pbar.close()
                # print("step_num",step_num)

                # collect data for this round
                all_video_paths[this_global_slice] = env.render()[this_local_slice]
                all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # import pdb; pdb.set_trace()
        # print(env.statelist[0])
        # print("env.statelist[0]",env.statelist[0])
        # print("env.statelist[0][0][0][-1][pos_agent]",env.statelist[0][0][0][-1]['pos_agent'])
        # print("len(env.statelist)",len(env.statelist)) # 56
        # print("len(env.statelist[0][0])",len(env.statelist[0][0])) # 108
        # print("len(env.statelist[0][0][0])",len(env.statelist[0][0][0])) # 4 是一个tuple，有一个pos_agent
        # # print("type",type(env.statelist[0][0][0]))
        # count_pos_agent = sum(1 for state in env.statelist[0][0] if isinstance(state[-1], dict) and 'pos_agent' in state[-1])
        # # print(len(env.statelist[0]['pos_agent']))
        # print("count",count_pos_agent)
        # traj = env.statelist[0][-1]
        # log
        # print(env.statelist)
        # for i, traj in enumerate(env.statelist):
            # Extract the points in the trajectory
            # trajectory_points = [point[-1]['pos_agent'] for point in traj[0]]
            # trajectory_points = np.array(trajectory_points)
            
            # # Plot the trajectory
            # plt.figure()  # Create a new figure for each trajectory
            # plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], marker='o', markersize=3)
            
            # # Set the title and labels
            # plt.title(f'Trajectory {i+1}')
            # plt.xlabel('X Position')
            # plt.ylabel('Y Position')
            # plt.grid(True)
            
            # # Save the figure
            # plt.savefig(f'trajectory_{i+1}.png')
            
            # # Optionally, display the plot
            # # plt.show()
            
            # # Close the plot to avoid overlapping of plots in the next iteration
            # plt.close()
            # output_directory = 'replay_data'
            # if not os.path.exists(output_directory):
            #     os.makedirs(output_directory)

            # with h5py.File(os.path.join(output_directory, f'replay_{i+1}.hdf5'), 'w') as f:
            #     # Extract and store observations
            #     obs = [point[0] for point in traj[0]]
            #     print()
            #     f.create_dataset('obs', data=np.array(obs))
                
            #     # Extract and store rewards
            #     rewards = [point[1] for point in traj[0]]
            #     f.create_dataset('reward', data=np.array(rewards))
                
            #     dones = [point[2] for point in traj[0]]
            #     f.create_dataset('done', data=np.array(dones, dtype=np.bool_))  # Ensure dtype is correct

            #     # Extract and store additional info
            #     for point in traj[0]:
            #         print(point)
            #     pos_agent = [point[3]['pos_agent'] for point in traj[0]]
            #     f.create_dataset('info/pos_agent', data=np.array(pos_agent))

            #     vel_agent = [point[3]['vel_agent'] for point in traj[0]]
            #     f.create_dataset('info/vel_agent', data=np.array(vel_agent))

            #     block_pose = [point[3]['block_pose'] for point in traj[0]]
            #     f.create_dataset('info/block_pose', data=np.array(block_pose))

            #     goal_pose = [point[3]['goal_pose'] for point in traj[0]]
            #     f.create_dataset('info/goal_pose', data=np.array(goal_pose))

            #     n_contacts = [point[3]['n_contacts'] for point in traj[0]]
            #     f.create_dataset('info/n_contacts', data=np.array(n_contacts))
                
            #     var_s = [point[4] for point in traj[0]]
            #     f.create_dataset('vars', data=np.array(var_s))
                
            #     actions = [point[-1] for point in traj[0]]
            #     f.create_dataset('actions', data=np.array(actions))
                
                # # Optionally, store environment state (if needed)
                # env_state = env.get_state()  # Example method to get env state
                # f.create_dataset('env_state', data=env_state)
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

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data