from typing import Callable
import snake_env
import gymnasium as gym
from stable_baselines3 import DQN
from cnn import CustomSnakeCNN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


if __name__ == '__main__':
    eval_env = gym.make('snake-env-v0', render_mode=None)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecFrameStack(eval_env, 8)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                                 log_path="./logs/", eval_freq=10000,
                                 deterministic=True, render=False)

    env = gym.make('snake-env-v0', render_mode=None)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 8)

    policy_kwargs = dict(
        features_extractor_class=CustomSnakeCNN,
        features_extractor_kwargs=dict(features_dim=3),
    )
    model = DQN("CnnPolicy", env, verbose=1,
                tensorboard_log="tensorboard/", exploration_fraction=0.07,
                policy_kwargs=policy_kwargs,
                seed=123, learning_starts=1000,
                learning_rate=linear_schedule(0.001))
    model.learn(total_timesteps=3000000,
                progress_bar=True, callback=eval_callback)
    model.save('snake-5')
