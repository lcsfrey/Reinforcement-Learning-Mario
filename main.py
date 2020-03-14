import time
import multiprocessing
import numpy as np

import torch
from mario_rl.dqn import DQN, compute_cost
from mario_rl.utils.logging import Logger
import os


RELOAD_PREVIOUS_MODEL = False    # Reloads model from previous training session
PROFILE_CODE = False             # Profile code line by line
MAX_STEPS = 30000                # Maximum number of iterations
LOG_ITERS = 100                  # Interval with which to log metrics
MAX_ITERS_NO_IMPROVEMENT = 200   # Maximum number of iterations before reseting game if no improvement
FRAMES_PER_SECOND = 30           # Number of frames per second to display
MAX_FRAMES_PER_SECOND = 30
USE_WANDB = False
USE_MULTIPROCESSING = False
OUTPUT_DIR = None  # 'F:\\wandb_mario_experiments'

def play_mario(*args, **kwargs):
    from mario_rl.history import History
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    RAW_IMAGE_SIZE = env.observation_space.shape
    # Create DQAN model
    history = History()
    model = DQN(input_size=RAW_IMAGE_SIZE, 
                nb_action=env.action_space.n, 
                gamma=0.98, 
                lrate=0.01, 
                history=history, 
                reload_previous_model=RELOAD_PREVIOUS_MODEL)

    logger = Logger(log_dir=OUTPUT_DIR, use_wandb=USE_WANDB)

    done = True
    reward = None
    start_time = time.time()
    for current_step in range(1, MAX_STEPS):
        start_of_step_time = time.time()
        if done:
            history["x_pos"] = None
            history["prev_x_pos"] = None
            history["y_pos"] = None
            history["prev_y_pos"] = None
            history["overall_reward"] = 0.0
            print(f"Best improvement so far: {history.overall_best}")
            print(f"Average improvement so far: {history.recent_best}")
            print(f"Last improvement: {reward}")
            print(f"Current step: {current_step}/{MAX_STEPS}")
            state = env.reset()
            history.reset()

        action = model.select_action(state)
        action = action.cpu().numpy().item()

        # TODO: Get sparse attention maps from model for visualization
        
        # take a step in the game
        next_state, raw_reward, done, info = env.step(action)

        # determine how good the move was
        # TODO: Incorporate learned cost function via actor/critic model
        # TODO: Create subgoals to try to maximize global goals.
        #       If those subgoals ddon't improve upon global goals, 
        #       throw away and create new subgoals
        #       Add relavent subgoals to memory
        current_reward, overall_reward, movement = compute_cost(
            env_reward=raw_reward, overall_reward=history["overall_reward"],
            x_pos=info["x_pos"], prev_x_pos=history["prev_x_pos"],
            y_pos=info["y_pos"], prev_y_pos=history["prev_y_pos"],
            done=done)

        # Add all important information about current iteration to history
        history.update(
            current_step=current_step,
            prev_reward=overall_reward,
            current_reward=current_reward,
            reward=overall_reward,
            overall_reward=overall_reward,
            raw_reward=raw_reward,
            action=action,
            movement=movement,
            action_rate=model.action_rate, 
            done=done,
            prev_x_pos=info["x_pos"],
            prev_y_pos=info["y_pos"],
        )

        # model ocasionally trains for a few generations
        model.update(state, action, overall_reward, next_state, done)

        # log infomation every LOG_ITERS actions made
        if done or model.action_count % LOG_ITERS:
            logger.log_metrics(next_state, info, history)

        state = next_state

        # TODO: Add early stopping if no improvement is seen for K iterations

        end_of_step_time = time.time()
        step_time_taken = (end_of_step_time - start_of_step_time)
        current_fps = 1/step_time_taken
        if current_fps > FRAMES_PER_SECOND:
            env.render()
        
        time.sleep(max(0, step_time_taken - 1/MAX_FRAMES_PER_SECOND))

    end_time = time.time()
    logger.log({"total_time(seconds)": end_time - start_time})
    time.sleep(torch.empty(1).random_(to=10).item())

    # TODO: Average the weights of multiple runs together
    model.save()
    env.close()
    logger.close()

def main(profile=False, use_multiprocessing=False):

    if profile:
        from line_profiler import LineProfiler
        profiler = LineProfiler()
        profiler.add_function(DQN.update)
        profiler.add_function(DQN.learn)
        profiler.add_function(DQN.select_action)
        #play_mario = profiler(play_mario)

    if use_multiprocessing:
        with multiprocessing.Pool(4) as pool:
            pool.map(play_mario, [None]*4)
    else:
        play_mario()

    if profile:
        profiler.print_stats()

if __name__ == "__main__":
    main(profile=PROFILE_CODE, use_multiprocessing=USE_MULTIPROCESSING)