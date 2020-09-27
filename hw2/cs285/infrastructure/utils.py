import math

import numpy as np
import time
import copy
from multiprocessing import Value

from cs285.infrastructure.multi_processing import run_multiprocessing_tasks

############################################
############################################

def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states

def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def mean_squared_error(a, b):
    return np.mean((a-b)**2)

############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    # TODO: get this from hw1

    # initialize env for the beginning of a new rollout
    # https://gym.openai.com/docs/#environments
    ob = env.reset() # HINT: should be the output of resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob) # HINT: query the policy's get_action function
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # HINT: rollout can end due to done, or due to max_path_length
         # HINT: this is either 0 or 1
        if (done or steps >= max_path_length):
            rollout_done = 1
        else:
            rollout_done = 0

        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def env_step(
    env,
    action,
    rollout_done=False,
    render=False,
    render_mode=('rgb_array'),
):
    image_ob = None
    reward = None
    next_observation = None
    done = None

    if rollout_done:
        return image_ob, reward, next_observation, done

    # render image of the simulated env
    if render:
        if 'rgb_array' in render_mode:
            if hasattr(env, 'sim'):
                image_ob = env.sim.render(camera_name='track', height=500, width=500)[::-1]
            else:
                image_ob = env.render(mode=render_mode)
        if 'human' in render_mode:
            env.render(mode=render_mode)
            time.sleep(env.model.opt.timestep)

    # take that action and record results
    next_observation, reward, done, _ = env.step(action)

    return image_ob, reward, next_observation, done

def env_step_in_one_thread(
    tasks,
    render=False,
    render_mode=('rgb_array'),
):
    assert len(tasks) == 1
    # TODO: env should be shared
    results = []
    for t in tasks:
        results.append((tasks[0]['env'],) + env_step(
            env=tasks[0]['env'],
            action=tasks[0]['action'],
            rollout_done=tasks[0]['rollout_done'],
            render=render,
            render_mode=render_mode,
        ))
    return results

def sample_trajectories_batch(
    batch_envs,
    policy,
    max_path_length,
    render=False,
    render_mode=('rgb_array'),
):
    paths = []
    batch_data = []
    batch_last_observations = []

    for i, env in enumerate(batch_envs):
        # init vars
        batch_data.append({
            "observations": [],
            "image_obs": [],
            "rewards": [],
            "actions": [],
            "next_observations": [],
            "terminals": [],
        })
        # initialize env for the beginning of a new rollout
        # https://gym.openai.com/docs/#environments
        batch_last_observations.append(env.reset())

    batch_last_observations = np.array(batch_last_observations)
    rollout_done_indices = set()
    steps = 0

    while True:
        steps += 1

        # use the most recent ob to decide what to do
        # batch_new_actions: (batch_size, action_dim)
        batch_new_actions = policy.get_action(batch_last_observations)

        for i, env in enumerate(batch_envs):

            ac = batch_new_actions[i]
            image_ob, rew, ob, done = env_step(
                env=env,
                action=ac,
                rollout_done=(i in rollout_done_indices),
                render=render,
                render_mode=render_mode,
            )

            if rew is None:
                continue

            # record result of taking that action
            batch_data[i]['observations'].append(batch_last_observations[i])
            if image_ob is not None:
                batch_data[i]['image_obs'].append(image_ob)
            batch_data[i]['actions'].append(ac)
            batch_data[i]['next_observations'].append(ob)
            batch_data[i]['rewards'].append(rew)
            batch_last_observations[i] = ob

            # HINT: rollout can end due to done, or due to max_path_length
             # HINT: this is either 0 or 1
            if (done or steps >= max_path_length):
                rollout_done = 1
                rollout_done_indices.add(i)
            else:
                rollout_done = 0

            batch_data[i]['terminals'].append(rollout_done)

        if len(rollout_done_indices) >= len(batch_envs):
            break

    for i, data in enumerate(batch_data):
        paths.append(Path(
            obs=data['observations'],
            image_obs=data['image_obs'],
            acs=data['actions'],
            rewards=data['rewards'],
            next_obs=data['next_observations'],
            terminals=data['terminals'],
        ))

    return paths

def sample_trajectories_batch_mp(
    batch_envs,
    policy,
    max_path_length,
    render=False,
    render_mode=('rgb_array'),
):
    paths = []
    batch_data = []
    batch_last_observations = []

    for i, env in enumerate(batch_envs):
        # init vars
        batch_data.append({
            "observations": [],
            "image_obs": [],
            "rewards": [],
            "actions": [],
            "next_observations": [],
            "terminals": [],
        })
        # initialize env for the beginning of a new rollout
        # https://gym.openai.com/docs/#environments
        batch_last_observations.append(env.reset())

    batch_last_observations = np.array(batch_last_observations)
    rollout_done_indices = set()
    steps = 0

    while True:
        steps += 1

        # use the most recent ob to decide what to do
        # batch_new_actions: (batch_size, action_dim)
        batch_new_actions = policy.get_action(batch_last_observations)
        tasks = [{
            'env': batch_envs[i],
            'action': batch_new_actions[i],
            'rollout_done': i in rollout_done_indices
        } for i in range(len(batch_envs))]
        results = run_multiprocessing_tasks(
            tasks=tasks,
            thread_func=env_step_in_one_thread,
            func_args=(
                render,
                render_mode
            ),
            num_cores=len(batch_envs),
            join_results=False
        )

        for i in range(len(batch_envs)):

            ac = batch_new_actions[i]
            assert len(results[i]) == 1
            (env, image_ob, rew, ob, done) = results[i][0]

            if rew is None:
                continue

            # record result of taking that action
            batch_envs[i] = env

            batch_data[i]['observations'].append(batch_last_observations[i])
            if image_ob is not None:
                batch_data[i]['image_obs'].append(image_ob)
            batch_data[i]['actions'].append(ac)
            batch_data[i]['next_observations'].append(ob)
            batch_data[i]['rewards'].append(rew)

            batch_last_observations[i] = ob

            # HINT: rollout can end due to done, or due to max_path_length
             # HINT: this is either 0 or 1
            if (done or steps >= max_path_length):
                rollout_done = 1
                rollout_done_indices.add(i)
            else:
                rollout_done = 0

            batch_data[i]['terminals'].append(rollout_done)

        if len(rollout_done_indices) >= len(batch_envs):
            break

    for i, data in enumerate(batch_data):
        paths.append(Path(
            obs=data['observations'],
            image_obs=data['image_obs'],
            acs=data['actions'],
            rewards=data['rewards'],
            next_obs=data['next_observations'],
            terminals=data['terminals'],
        ))

    return paths


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    # TODO: get this from hw1

    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    if isinstance(env, list) or isinstance(env, tuple):
        while timesteps_this_batch < min_timesteps_per_batch:
            new_paths = sample_trajectories_batch(
                batch_envs=env,
                policy=policy,
                max_path_length=max_path_length,
                render=render,
                render_mode=render_mode,
            )
            paths.extend(new_paths)
            timesteps_this_batch += sum([get_pathlength(p) for p in new_paths])
    else:
        while timesteps_this_batch < min_timesteps_per_batch:
            paths.append(sample_trajectory(
                env=env,
                policy=policy,
                max_path_length=max_path_length,
                render=render,
                render_mode=render_mode,
            ))
            timesteps_this_batch += get_pathlength(paths[-1])

    return paths, timesteps_this_batch

def sample_trajectories_mp(
    env,
    policy,
    min_timesteps_per_batch,
    max_path_length,
    render=False,
    render_mode=('rgb_array'),
    num_cores=4
):
    min_timesteps_per_thread = math.ceil(min_timesteps_per_batch/num_cores)
    (paths, timesteps_this_batch) = run_multiprocessing_tasks(
        tasks=[],
        thread_func=sample_trajectories,
        func_args=(
            env,
            policy,
            min_timesteps_per_thread,
            max_path_length,
            render,
            render_mode
        ),
        num_cores=num_cores,
        join_results=True
    )
    paths = sum(paths, [])
    timesteps_this_batch = sum(timesteps_this_batch)
    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    # TODO: get this from hw1

    """
        Collect ntraj rollouts.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    """
    paths = []
    for _ in range(ntraj):
        paths.append(sample_trajectory(
            env=env,
            policy=policy,
            max_path_length=max_path_length,
            render=render,
            render_mode=render_mode,
        ))


    return paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data