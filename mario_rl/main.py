import time
import multiprocessing
import numpy as np
from dqn import DQN
import os

try:
    import wandb
    wandb.init(project="mario-rl")
    USE_WANDB = True
except ImportError:
    USE_WANDB = False
    print("ERROR: wandB not installed. Metrics will not be stored")

RELOAD_PREVIOUS_MODEL = False   # Reloads model from previous training session
PROFILE_CODE = False            # Profile code line by line
MAX_STEPS = 100000              # Maximum number of iterations
LOG_ITERS = 25                  # Interval with which to log metrics
MAX_ITERS_NO_IMPROVEMENT = 200  # Maximum number of iterations before reseting game if no improvement
ITERS_BEFORE_LEARNING = 100     # Steps before between model updates
REPEAT_ITERS_MEAN = 15.0        # Mean time spent applying an action
REPEAT_ITERS_STD = 4.0          # Standard deviation of time spent applying an action
FRAMES_PER_SECOND = 30          # Number of frames per second to display
MAX_FRAMES_PER_SECOND = 30

if USE_WANDB:
    wandb.log({"REPEAT_ITERS_MEAN": REPEAT_ITERS_MEAN,
               "MAX_STEPS": MAX_STEPS})

def compute_cost(env_reward, x_pos, prev_x_pos=None, done=False):
    movement = 0
    if done:
        reward = -15
    elif prev_x_pos is not None:
        movement = x_pos - prev_x_pos

        reward = movement * abs(env_reward)
        reward = -1 if movement == 0 else reward
    else:
        reward = env_reward

    return reward, movement


def log_metrics(state, info, history):
    if not USE_WANDB:
        return

    if history["done"]:
        image_caption = "step: {} action: {}".format(
             history['current_step'], history['action'])

        wandb.log(
            {"frame": [wandb.Image(state, caption=image_caption)]}, 
            step=history['current_step'], 
            commit=False
        )

    wandb.log({**info, **history}, step=history['current_step'])

def play_mario(*args, **kwargs):
    import torch
    from history import History
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    RAW_IMAGE_SIZE = env.observation_space.shape
    REPEAT_ITERS_SAMPLER = torch.distributions.normal.Normal(
        loc=REPEAT_ITERS_MEAN, scale=REPEAT_ITERS_STD)
    # Create DQAN model
    history = History()
    model = DQN(input_size=RAW_IMAGE_SIZE, 
                nb_action=env.action_space.n, 
                gamma=0.98, 
                lrate=0.01, 
                history=history)

    if RELOAD_PREVIOUS_MODEL:
        try:
            model.load()
        except ValueError:
            print("Model loading failed")

    if USE_WANDB:
        # Log metrics with wandb
        wandb.watch(model)

    done = True
    reward = None
    action_repeat_time = REPEAT_ITERS_MEAN
    actions_made = 0
    start_time = time.time()
    for current_step in range(1, MAX_STEPS):
        start_of_step_time = time.time()
        if done:
            history.prev_x_pos = None
            print(f"Best improvement so far: {history.overall_best}")
            print(f"Average improvement so far: {history.recent_best}")
            print(f"Last improvement: {reward}")
            print(f"Current step: {current_step}/{MAX_STEPS}")
            state = env.reset()
            history.reset()
            torch.cuda.empty_cache()

        if done or current_step % action_repeat_time == 0:
            with torch.no_grad():
                action = model.select_action(state).cpu().numpy().item()
            actions_made += 1

        # TODO: Get sparse attention maps from model for visualization
        
        # take a step in the game
        next_state, raw_reward, done, info = env.step(action)

        # determine how good the move was
        # TODO: Incorporate learned cost function via actor/critic model
        # TODO: Create subgoals to try to maximize global goals.
        #       If those subgoals ddon't improve upon global goals, 
        #       throw away and create new subgoals
        #       Add relavent subgoals to memory
        reward, movement = compute_cost(env_reward=raw_reward,
                                        x_pos=info["x_pos"], 
                                        prev_x_pos=history.prev_x_pos,
                                        done=done)

        # Add all important information about current iteration to history
        history.update(
            current_step=current_step,
            reward=reward,
            raw_reward=raw_reward,
            action=action,
            movement=movement,
            action_repeat_time=action_repeat_time, 
            done=done
        )
        # model ocasionally trains for a few generations
        model.update(state, action, reward, next_state, done)

        if current_step % action_repeat_time == 0:
            if actions_made % LOG_ITERS:
                # log infomation every LOG_ITERS actions made
                log_metrics(state, info, history)

            # pick a new amount of time to take the next action
            action_repeat_time = (
                REPEAT_ITERS_SAMPLER
                .sample()
                .clamp_(min=1)
                .round_()
                .int()
                .item()
            )

        state = next_state
        history.prev_x_pos = info["x_pos"]

        # TODO: Add early stopping if no improvement is seen for K iterations

        end_of_step_time = time.time()
        step_time_taken = (end_of_step_time - start_of_step_time)
        current_fps = 1/step_time_taken
        if current_fps > FRAMES_PER_SECOND:
            env.render()
        
        time.sleep(max(0, step_time_taken - 1/MAX_FRAMES_PER_SECOND))

    end_time = time.time()

    model.save()
    env.close()
    if USE_WANDB:
        wandb.log({"total_time(seconds)": end_time - start_time})
        # Save model to wandb
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    time.sleep(torch.empty(1).random_(to=20).item())

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
    main(profile=PROFILE_CODE)