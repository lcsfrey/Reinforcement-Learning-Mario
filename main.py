import time
import multiprocessing
import numpy as np

import torch
from mario_rl.dqn import DQN
from mario_rl.utils.logging import Logger
import os

import gym

RELOAD_PREVIOUS_MODEL = False    # Reloads model from previous training session
PROFILE_CODE = False             # Profile code line by line
MAX_STEPS = 50000                # Maximum number of iterations
LOG_ITERS = 100                  # Interval with which to log metrics
MAX_ITERS_NO_IMPROVEMENT = 100   # Maximum number of iterations before reseting game if no improvement
FRAMES_PER_SECOND =  100         # Number of frames per second to display
USE_WANDB = True                 # If True, logs will be recorded using Weights & Biases
USE_MULTIPROCESSING = False      # If True, N parallel games will be trained
NUM_PARALLEL_GAMES = 4           # Number of games played in parallel when using multiprocessing
OUTPUT_DIR = './output/wandb_mario_experiments'  # output directory to save logs

def play_mario(*args, **kwargs):
    from mario_rl.history import History
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Create DQN model
    model = DQN(input_size=env.observation_space.shape, 
                nb_action=env.action_space.n, 
                gamma=0.9, 
                lrate=0.001,
                temperature=0.95,
                reload_previous_model=RELOAD_PREVIOUS_MODEL,
                store_feature_maps=True)

    logger = Logger(env=env, model=model, 
                    log_dir=OUTPUT_DIR, use_wandb=USE_WANDB)
    history = History(logger=logger, log_iters=LOG_ITERS)

    def step(model, state, current_step, total_current_reward):
        # get the next action from the model
        action = model.select_action(state).item()

        # TODO: get sparse attention maps from model for visualization
        
        # take a step in the game
        next_state, reward, done, info = env.step(action)

        total_current_reward += reward

        # update the model
        loss = model.update(state, action, reward, next_state, done)

        # Add all important information about current iteration to history
        history.update(
            step_num=current_step,
            force_log=done,
            action=action,
            state=state,
            action_rate=model.action_rate,
            reward=reward,
            total_current_reward=total_current_reward+reward,
            done=done,
            loss=loss,
            **{k: v.flatten() for k, v in model.feature_maps.items()},
        )

        return model, next_state, reward, total_current_reward, done
        
    start_time = time.time()
    time_since_last_frame = time.time()
    state = env.reset()
    no_improvement_cntr = 0
    total_current_reward = 0
    for current_step in range(MAX_STEPS):

        model, state, reward, total_current_reward, done \
            = step(model, state, current_step, total_current_reward)

        # check if any improvement has been made
        no_improvement_cntr = 0 if reward > 0 else no_improvement_cntr + 1
        if no_improvement_cntr > MAX_ITERS_NO_IMPROVEMENT:
            no_improvement_cntr = 0
            done = True

        current_fps = 1/(time.time() - time_since_last_frame)
        if current_fps < FRAMES_PER_SECOND:
            env.render()
            time_since_last_frame = time.time()

        if done:
            print(f"Best improvement so far: {history.overall_best}")
            print(f"Average improvement so far: {history.recent_best}")
            print(f"Last improvement: {reward}")
            print(f"Current step: {current_step}/{MAX_STEPS}")
            state = env.reset()
            history.reset()
            total_current_reward = 0

    end_time = time.time()
    logger.log({"total_time(seconds)": end_time - start_time})

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
        with multiprocessing.Pool(NUM_PARALLEL_GAMES) as pool:
            pool.map(play_mario, [None]*NUM_PARALLEL_GAMES)
    else:
        play_mario()

    if profile:
        profiler.print_stats()

if __name__ == "__main__":
    main(profile=PROFILE_CODE, use_multiprocessing=USE_MULTIPROCESSING)