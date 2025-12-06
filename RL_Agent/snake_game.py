# snake_game.py
import pygame
import random
import numpy as np

BLOCK_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
SCREEN_WIDTH = GRID_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * BLOCK_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class SnakeGame:
    def __init__(self, headless=False):
        self.headless = headless

        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake AI")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.direction = RIGHT
        self.snake = [(5, 5)]
        self.food = self._place_food()
        self.score = 0
        self.frame_iteration = 0
        return self.get_state()

    def step(self, action):
        self.frame_iteration += 1
        self._move(action)
        reward = 0
        done = False

        # Collision
        if self._is_collision():
            reward = -50
            done = True
            return self.get_state(), reward, done, {}

        # Eat food
        if self.snake[0] == self.food:
            self.score += 1
            reward = 10
            self.snake.append(self.snake[-1])
            self.food = self._place_food()
        else:
            reward = 1
            self.snake.pop()

        # Optional: step limit to prevent infinite loops
        if self.frame_iteration > 500:
            done = True
            reward = -10

        return self.get_state(), reward, done, {}

    def render(self):
        if self.headless:
            return  # Do nothing → FAST evaluation

        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, RED, pygame.Rect(
            self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        for block in self.snake:
            pygame.draw.rect(self.screen, GREEN, pygame.Rect(
                block[0]*BLOCK_SIZE, block[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        pygame.display.flip()
        self.clock.tick(10)

    def get_state_size(self):
        return 11

    def get_action_size(self):
        return 3

    def _place_food(self):
        while True:
            pos = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            if pos not in self.snake:
                return pos

    def _move(self, action):
        clock_wise = [RIGHT, DOWN, LEFT, UP]

        # ---- SAFETY PATCH ----
        try:
            idx = clock_wise.index(self.direction)
        except ValueError:
            self.direction = RIGHT
            idx = 0
        # ----------------------

        # Ensure action is safe
        action = np.array(action).flatten()
        if action.shape != (3,) or np.sum(action) != 1:
            action = np.array([1,0,0])  # fallback to straight

        # action = [straight, right, left]
        if np.array_equal(action, [1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0,1,0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x = self.snake[0][0] + self.direction[0]
        y = self.snake[0][1] + self.direction[1]
        self.snake.insert(0, (x, y))

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.snake[0]

        if pt[0] < 0 or pt[0] >= GRID_WIDTH or pt[1] < 0 or pt[1] >= GRID_HEIGHT:
            return True

        if pt in self.snake[1:]:
            return True

        return False

    def get_state(self):
        head = self.snake[0]
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)

        dir_l = self.direction == LEFT
        dir_r = self.direction == RIGHT
        dir_u = self.direction == UP
        dir_d = self.direction == DOWN

        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),

            # Danger right
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),

            # Danger left
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),

            # Food location
            self.food[0] < head[0],
            self.food[0] > head[0],
            self.food[1] < head[1],
            self.food[1] > head[1],

            # Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d
        ]

        return np.array(state, dtype=int)
