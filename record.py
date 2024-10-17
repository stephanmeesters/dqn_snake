import snake_env
import gymnasium as gym
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from matplotlib.animation import FuncAnimation


def update(frame):
    global obs, img1, img2, model, vec_env
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)

    img1 = env.unwrapped.env_method('get_grid')[0]
    img2 = np.rot90(obs[0, 0, :, :], k=1)

    img_plot1.set_data(img1)
    img_plot2.set_data(img2)
    return [img_plot1, img_plot2]


if __name__ == '__main__':
    env = gym.make('snake-env-v0', render_mode=None)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 8)

    model = DQN.load("pretrained_model/best_model", env=env)
    vec_env = model.get_env()
    obs = vec_env.reset()

    img1 = env.unwrapped.env_method('get_grid')[0]
    img2 = obs[0, 0, :, :]

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    img_plot1 = ax1.imshow(img1, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('grid view')
    img_plot2 = ax2.imshow(img2, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('snake view')

    ani = FuncAnimation(fig, update, frames=range(1000),
                        blit=True, interval=50)

    ani.save('snake_animation.gif', writer='imagemagick')

    plt.show()

