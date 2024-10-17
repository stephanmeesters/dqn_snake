import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
from enum import Enum
import copy


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class ChangeDirectionAction(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2


class GridPoint(Enum):
    NONE = 0
    SNAKE = 255
    BARRIER = 255


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def toarr(self):
        return np.array([self.x, self.y])


class Agent:
    def __init__(self, team_id, position, direction):
        self.team_id = team_id
        self.direction = direction
        self.alive = True
        self.position = position


def make_move(pos, direction):
    newpos = Position(pos.x, pos.y)
    if direction == Direction.UP:
        newpos.y += 1
    elif direction == Direction.DOWN:
        newpos.y -= 1
    elif direction == Direction.LEFT:
        newpos.x -= 1
    elif direction == Direction.RIGHT:
        newpos.x += 1
    return newpos


def change_direction(direction, change):
    if direction == Direction.UP:
        if change == ChangeDirectionAction.NONE:
            return Direction.UP
        elif change == ChangeDirectionAction.LEFT:
            return Direction.LEFT
        elif change == ChangeDirectionAction.RIGHT:
            return Direction.RIGHT
    elif direction == Direction.DOWN:
        if change == ChangeDirectionAction.NONE:
            return Direction.DOWN
        elif change == ChangeDirectionAction.LEFT:
            return Direction.RIGHT
        elif change == ChangeDirectionAction.RIGHT:
            return Direction.LEFT
    elif direction == Direction.LEFT:
        if change == ChangeDirectionAction.NONE:
            return Direction.LEFT
        elif change == ChangeDirectionAction.LEFT:
            return Direction.DOWN
        elif change == ChangeDirectionAction.RIGHT:
            return Direction.UP
    elif direction == Direction.RIGHT:
        if change == ChangeDirectionAction.NONE:
            return Direction.RIGHT
        elif change == ChangeDirectionAction.LEFT:
            return Direction.UP
        elif change == ChangeDirectionAction.RIGHT:
            return Direction.DOWN


class SnakeGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.full(
            (width, height), GridPoint.NONE.value, dtype=np.uint8)

    def clone(self):
        return copy.deepcopy(self)


def position_is_legal(position, grid):
    return (0 <= position.x < grid.width and
            0 <= position.y < grid.height and
            grid.grid[position.x, position.y] == GridPoint.NONE.value)


def find_legal_directions(position, direction, grid):
    moves = {}
    for cdirection in ChangeDirectionAction:
        new_direction = change_direction(direction, cdirection)
        new_pos = make_move(position, new_direction)
        if position_is_legal(new_pos, grid):
            moves[cdirection] = new_direction
    return moves


def direction_is_legal(newdir, olddir):
    if newdir == Direction.DOWN and olddir == Direction.UP:
        return False
    if newdir == Direction.LEFT and olddir == Direction.RIGHT:
        return False
    if newdir == Direction.RIGHT and olddir == Direction.LEFT:
        return False
    if newdir == Direction.UP and olddir == Direction.DOWN:
        return False
    return True


def gen_random_grid(width, height, barriers_percentage):
    agentgrid = SnakeGrid(width, height)
    num_gridpoints = (width - 3) * (height - 3)
    num_barriers = int(num_gridpoints * barriers_percentage)
    while num_barriers > 0:
        while True:
            x = np.random.randint(0, width - 1)
            y = np.random.randint(0, height - 1)
            if agentgrid.grid[x, y] == GridPoint.NONE.value:
                agentgrid.grid[x, y] = GridPoint.BARRIER.value
                break
        num_barriers = num_barriers - 1
    return agentgrid


register(id='snake-env-v0', entry_point='snake_env:SnakeEnv')


class SnakeEnv(gym.Env):

    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, start_grid=None, fov=10, render_mode=None):
        self.start_grid = gen_random_grid(50, 50, 0.1)
        self.fov = fov
        self.fullfov = fov * 2 + 1
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(len(ChangeDirectionAction))
        self.observation_space = spaces.Box(low=0, high=255, shape=(
            1, self.fullfov, self.fullfov), dtype=np.uint8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)

        if self.start_grid is None:
            self.grid = gen_random_grid(50, 50, 0.1)
        else:
            self.grid = self.start_grid.clone()
        self.agent = None
        self.done = False

        while True:
            x = np.random.randint(0, self.grid.width - 1)
            y = np.random.randint(0, self.grid.height - 1)
            if self.grid.grid[x, y] == GridPoint.NONE.value:
                break
        direction = np.random.choice(list(Direction))
        team_id = np.random.randint(0, 10)
        position = Position(x, y)
        self.agent = Agent(team_id, position, direction)
        self.grid.grid[x, y] = GridPoint.SNAKE.value

        if self.render_mode == 'human':
            self.render()

        obs = self._get_observation()
        info = {}

        return obs, info

    def sample(self):
        return self.step(np.random.choice(list(Direction)).value)

    def get_grid(self):
        return self.grid.grid

    def step(self, actions):
        reward = 0
        infos = {}

        agent = self.agent
        action = actions
        if agent.alive:
            change = ChangeDirectionAction(action)
            if change != ChangeDirectionAction.NONE:
                reward -= 0.2
            agent.direction = change_direction(agent.direction, change)

            agent.position = make_move(agent.position, agent.direction)

            if position_is_legal(agent.position, self.grid):
                self.grid.grid[agent.position.x,
                               agent.position.y] = GridPoint.SNAKE.value
            else:
                agent.alive = False
                reward = -100  # Negative reward for dying

        self.done = not agent.alive
        obs = self._get_observation()
        if agent.alive:
            reward += 1  # Small reward for staying alive
        infos['position'] = agent.position

        if self.render_mode == 'human':
            self.render()

        truncated = False
        return obs, reward, self.done, truncated, infos

    def render(self, mode='human'):
        grid_repr = np.full(
            (self.grid.width, self.grid.height), '.', dtype=str)
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid.grid[x, y] == GridPoint.SNAKE.value:
                    grid_repr[x, y] = 'S'
                elif self.grid.grid[x, y] == GridPoint.BARRIER.value:
                    grid_repr[x, y] = 'X'
        if self.agent.alive:
            grid_repr[self.agent.position.x, self.agent.position.y] = str(idx)
        print('\n'.join(''.join(row) for row in grid_repr.T))

    def _get_observation(self):
        fullfov = self.fov * 2 + 1
        screen = np.full((1, fullfov, fullfov),
                         GridPoint.BARRIER.value, dtype=np.uint8)

        agent = self.agent
        xmin = max(agent.position.x - self.fov, 0)
        xmax = min(agent.position.x + self.fov, self.grid.width - 1)
        ymin = max(agent.position.y - self.fov, 0)
        ymax = min(agent.position.y + self.fov, self.grid.height - 1)

        screenxmin = 0
        screenxmax = fullfov - 1
        screenymin = 0
        screenymax = fullfov - 1

        if xmin == 0:
            screenxmin = fullfov - (xmax - xmin) - 1
        elif xmax == self.grid.width - 1:
            screenxmax = (xmax - xmin)
        if ymin == 0:
            screenymin = fullfov - (ymax - ymin) - 1
        elif ymax == self.grid.height - 1:
            screenymax = (ymax - ymin)

        screen[0, screenxmin:screenxmax+1, screenymin:screenymax+1] = \
            np.expand_dims(
                self.grid.grid[xmin:xmax+1, ymin:ymax+1], axis=0)

        if agent.direction == Direction.RIGHT:
            screen[0, :, :] = np.rot90(screen[0, :, :], k=1, axes=(0, 1))
        elif agent.direction == Direction.DOWN:
            screen[0, :, :] = np.rot90(screen[0, :, :], k=2, axes=(0, 1))
        elif agent.direction == Direction.LEFT:
            screen[0, :, :] = np.rot90(screen[0, :, :], k=3, axes=(0, 1))

        return screen


if __name__ == '__main__':
    env = gym.make('snake-env-v0', render_mode='human')
    check_env(env.unwrapped)

    obs = env.reset()[0]

    for i in range(10):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)
