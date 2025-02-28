
import gymnasium as gym
import cv2
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

class TrackEdgeWrapper(gym.Wrapper):
    def __init__(self, env):
        """初始化包裝器，用於分割賽道邊線"""
        super(TrackEdgeWrapper, self).__init__(env)
        self.lower_white = np.array([0, 0, 200])  # HSV 下限
        self.upper_white = np.array([180, 25, 255])  # HSV 上限

    def process_image(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        segmented = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite("segmented_image.png", cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))  # 保存调试图像
        return segmented

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.process_image(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.process_image(observation), reward, done, info

def train_donkeycar():
    """訓練 DonkeyCar，使用 SAC 和 TrackEdgeWrapper"""
    # 創建帶有包裝器的環境
    env_id = "donkey-mountain-track-v0"  # 替換為實際環境名稱
    env = make_vec_env(
        env_id,
        n_envs=1,  # 單一環境，視硬體資源可增加
        wrapper_class=TrackEdgeWrapper  # 應用自定義包裝器
    )

    # 定義 SAC 模型，使用 CNN 處理圖像輸入
    model = SAC(
        "CnnPolicy",  # 使用 CNN 策略
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"  # 日誌路徑
    )

    # 訓練模型
    model.learn(total_timesteps=100000)

    # 保存模型
    model.save("sac_donkeycar_track_edges")

    # 關閉環境
    env.close()

if __name__ == "__main__":
    train_donkeycar()