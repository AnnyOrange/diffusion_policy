import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])

def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result

def aggregate(data, method='max'):
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps
        self.statelist = env.statelist
        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
        # print("env",env)
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        obs = self._get_obs(self.n_obs_steps)
        return obs
    ## 这个是快速版本的，但是没有加变色的
    # def process_action_and_var2x(self, action_and_var):
    #     # 提取 action_s, var_s 和 scale
    #     action_s = action_and_var[:, :2]    # (N, 2)
    #     var_s = action_and_var[:, 2:4]      # (N, 2)
    #     scale = action_and_var[:, 5]        # (N,)
        
    #     # 第一步：计算 var_s 的平均值
    #     avg_var_s = np.mean(var_s, axis=1)  # 计算每一行的平均值，形状为 (N,)
        
    #     # 矢量化条件判断，找到需要删除的索引
    #     # 创建一个布尔数组，表示每个位置是否需要删除
    #     # 首先扩展 scale，以便与 avg_var_s 对齐
    #     scale_shifted = scale[1:]  # 向后移动一位
    #     avg_var_s_shifted = avg_var_s[1:]  # 向后移动一位
        
    #     # 比较条件
    #     condition = (avg_var_s[:-1] > scale[:-1]) & (avg_var_s_shifted > scale_shifted)
        
    #     # 需要删除的索引是满足条件的第一个元素的索引
    #     indices_to_remove = np.where(condition)[0]
        
    #     # 构建保留的索引列表
    #     indices = np.arange(len(avg_var_s))
    #     # 删除需要移除的索引
    #     indices = np.delete(indices, indices_to_remove)
        
    #     # 提取需要保留的 action_s 和 var_s
    #     action_s_kept = action_s[indices]
    #     var_s_kept = var_s[indices]
        
    #     # 合并 action_s 和 var_s
    #     action_and_var_kept = np.concatenate((action_s_kept, var_s_kept), axis=-1)
        
    #     return action_and_var_kept
    def process_action_and_var2x(self,action_and_var):
        action_s = action_and_var[:,:2]
        var_s = action_and_var[:,2:4]
        scale = action_and_var[:,-1]
        avg_var_s = np.mean(var_s, axis=1)  # 计算每一行的平均值，形状为 (8,)
        ## 倍速
        # 初始化一个列表用于存储保留的 action_s 的索引
        indices_to_keep = []
        flags = []
        i = 0
        while i < len(avg_var_s):
            if i < len(avg_var_s) - 1 and avg_var_s[i] > scale[i] and avg_var_s[i + 1] > scale[i + 1]:
                # 如果连续两个 avg_var_s 元素大于 scale，删除 i，保留 i+1
                indices_to_keep.append(i + 1)
                flags.append(1)
                i += 2  # 跳过 i 和 i+1，进入下一步
            else:
                # 否则保留当前 i
                indices_to_keep.append(i)
                flags.append(0)
                i += 1
                
        action_s_kept = action_s[indices_to_keep]
        var_s_kept = var_s[indices_to_keep]
        flags = np.array(flags).reshape(-1, 1)
        return np.concatenate((action_s_kept,var_s_kept,flags), axis=-1)
        
    def step(self, action_and_var):
    # def step(self,action):
        """
        action_and_var 包含action var 和 scale
        """
        if action_and_var[0, -1] > 0:
            action_and_var = self.process_action_and_var2x(action_and_var)
        for combined in action_and_var:
        # for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            
            observation, reward, done, info = super().step(combined)
            # observation, reward, done, info = super().step(act)
            # print("-----------",self.statelist)
            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result