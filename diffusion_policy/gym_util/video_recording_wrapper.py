import gym
import numpy as np
from diffusion_policy.real_world.video_recorder import VideoRecorder
import cv2
class VideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            video_recoder: VideoRecorder,
            mode='rgb_array',
            file_path=None,
            steps_per_render=1,
            **kwargs
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.file_path = file_path
        self.video_recoder = video_recoder
        self.statelist = []
        self.env = env

        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        self.video_recoder.stop()
        return obs
    
    def step(self, action_var):
    # def step(self, action):
        # print("env",self.env.seed)
        # print("act_var.shape",action_var.shape)
        action = action_var[:2]
        # print("action",action)
        var = action_var[2:4]
        flag = action_var[-1]
        # v_mean_pos = (var[0]+var[1]+var[2])/3
        # v_mean_rot = (var[3]+var[4]+var[5])/3
        # v_mean_all = np.mean(var)
        # print("v_mean_pos",v_mean_pos)
        # print("v_mean_all",v_mean_all)
        # print(format(v_mean_all, "e"))
        v_mean = (var[0]+var[1])/2
        
        # import pdb; pdb.set_trace()

        
        # noise = 
        # if np.array_equal(if_noise, np.array([1.0, 1.0])):
        #     action = action+noise
        result = super().step(action)
        combined_result = result + (var,)+(action,)
        
        # print("result",combined_result)
        
        # print("result:", result, "action:", action)

        # print("result",result)
        self.step_count += 1
        self.statelist.append(combined_result)
        
        if self.file_path is not None \
            and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recoder.is_ready():
                self.video_recoder.start(self.file_path)
            # print(**self.render_kwargs)
            frame = self.env.render(
                mode=self.mode, **self.render_kwargs)
            assert frame.dtype == np.uint8
            if flag==0:
                frame = put_text_blue(frame,  f"{v_mean}")
            else:
                frame = put_text_red(frame,  f"{v_mean}")
            self.video_recoder.write_frame(frame)
        # print("------------------",self.statelist)
        return result
    
    def render(self, mode='rgb_array', **kwargs):
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        return self.file_path

def put_text_blue(img, text, is_waypoint=False, font_size=0.5, thickness=2, position="top"):
    img = img.copy()
    if position == "top":
        p = (10, 30)
    elif position == "bottom":
        p = (10, img.shape[0] - 60)
    # put the frame number in the top left corner
    img = cv2.putText(
        img,
        str(text),
        p,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    if is_waypoint:
        img = cv2.putText(
            img,
            "*",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
    return img
def put_text_red(img, text, is_waypoint=False, font_size=0.5, thickness=2, position="top"):
    img = img.copy()
    if position == "top":
        p = (10, 30)
    elif position == "bottom":
        p = (10, img.shape[0] - 60)
    # put the frame number in the top left corner
    img = cv2.putText(
        img,
        str(text),
        p,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    if is_waypoint:
        img = cv2.putText(
            img,
            "*",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
    return img

# def plot()