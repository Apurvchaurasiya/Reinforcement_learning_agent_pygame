from game import SnakeGameAI
from agent import PPOAgent
import numpy as np

def get_state(game):
    head = game.snake[0]

    # Points around head
    point_l = (head.x - 20, head.y)
    point_r = (head.x + 20, head.y)
    point_u = (head.x, head.y - 20)
    point_d = (head.x, head.y + 20)

    dir_l = game.direction.name == "LEFT"
    dir_r = game.direction.name == "RIGHT"
    dir_u = game.direction.name == "UP"
    dir_d = game.direction.name == "DOWN"

    state = [
        # Danger straight
        (dir_r and game.is_collision(point_r)) or
        (dir_l and game.is_collision(point_l)) or
        (dir_u and game.is_collision(point_u)) or
        (dir_d and game.is_collision(point_d)),

        # Danger right
        (dir_u and game.is_collision(point_r)) or
        (dir_d and game.is_collision(point_l)) or
        (dir_l and game.is_collision(point_u)) or
        (dir_r and game.is_collision(point_d)),

        # Danger left
        (dir_d and game.is_collision(point_r)) or
        (dir_u and game.is_collision(point_l)) or
        (dir_r and game.is_collision(point_u)) or
        (dir_l and game.is_collision(point_d)),

        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,

        # Food location
        game.food.x < head.x,
        game.food.x > head.x,
        game.food.y < head.y,
        game.food.y > head.y
    ]

    return np.array(state, dtype=int)


if __name__ == "__main__":
    game = SnakeGameAI()
    agent = PPOAgent(state_dim=11, action_dim=3)

    episode = 0

    while True:
        state = get_state(game)
        action = agent.select_action(state)

        move = [0, 0, 0]
        move[action] = 1

        reward, done, score = game.play_step(move)
        agent.store_reward(reward, done)

        if done:
            game.reset()
            agent.update()
            episode += 1
            print(f"Episode {episode}, Score {score}")
