import os
import shlex
import time
import argparse
import time
import itertools

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.explore_or_exploit_agent import ExplorationOrExploitationAgent
from cs285.infrastructure.dqn_utils import get_env_kwargs, PiecewiseSchedule, ConstantSchedule
from cs285.scripts.run_hw5_expl import Q_Trainer

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_name',
        default='PointmassHard-v0',
        choices=('PointmassEasy-v0', 'PointmassMedium-v0', 'PointmassHard-v0', 'PointmassVeryHard-v0')
    )

    parser.add_argument('--exp_name', type=str, default='todo')

    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--use_rnd', action='store_true')
    parser.add_argument('--num_exploration_steps', type=int, default=10000)
    parser.add_argument('--unsupervised_exploration', action='store_true')

    parser.add_argument('--offline_exploitation', action='store_true')
    parser.add_argument('--cql_alpha', type=float, default=0.0)

    parser.add_argument('--exploit_rew_shift', type=float, default=0.0)
    parser.add_argument('--exploit_rew_scale', type=float, default=1.0)

    parser.add_argument('--rnd_output_size', type=int, default=5)
    parser.add_argument('--rnd_n_layers', type=int, default=2)
    parser.add_argument('--rnd_size', type=int, default=400)

    # parameters for parallelization
    # on the same core, use num_envs_per_core as batch size to generate actions using policy
    parser.add_argument('--num_envs_per_core', type=int, default=1)
    # steps collected per eval iteration
    parser.add_argument('--num_cores', type=int, default=1)

    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=int(1e3))
    parser.add_argument('--save_params', action='store_true')
    return parser

def parse_command(cmd):
    cmd_parser = get_argument_parser()
    argument_list = shlex.split(cmd)
    argument_list = argument_list[2:]
    try:
        args = cmd_parser.parse_args(argument_list)
    except:
        raise AttributeError('Command parse error: \n{}\n'.format(cmd))
        # print(tmp_result['command'])
    return args

def train_w_parameters(cmd):
    args = parse_command(cmd)

    # convert to dictionary
    params = vars(args)
    params['double_q'] = True
    params['num_agent_train_steps_per_iter'] = 1
    params['num_critic_updates_per_agent_update'] = 1
    params['exploit_weight_schedule'] = ConstantSchedule(1.0)
    params['video_log_freq'] = -1  # This param is not used for DQN
    params['num_timesteps'] = 50000
    params['learning_starts'] = 2000
    params['eps'] = 0.2
    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if params['env_name'] == 'PointmassEasy-v0':
        params['ep_len'] = 50
    if params['env_name'] == 'PointmassMedium-v0':
        params['ep_len'] = 150
    if params['env_name'] == 'PointmassHard-v0':
        params['ep_len'] = 100
    if params['env_name'] == 'PointmassVeryHard-v0':
        params['ep_len'] = 200

    if params['use_rnd']:
        params['explore_weight_schedule'] = PiecewiseSchedule([(0, 1), (params['num_exploration_steps'], 0)],
                                                              outside_value=0.0)
    else:
        params['explore_weight_schedule'] = ConstantSchedule(0.0)

    if params['unsupervised_exploration']:
        params['explore_weight_schedule'] = ConstantSchedule(1.0)
        params['exploit_weight_schedule'] = ConstantSchedule(0.0)

        if not params['use_rnd']:
            params['learning_starts'] = params['num_exploration_steps']

    logdir_prefix = 'hw5_expl_'  # keep for autograder
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    trainer = Q_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    all_cmds = []
    os.chdir('../../')

    # # #############################################
    # # exp 1
    # # #############################################
    # # env
    # all_cmds.append('''
    #     python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --use_rnd
    #     --unsupervised_exploration --exp_name q1_env1_rnd
    # ''')

    # all_cmds.append('''
    #     python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0
    #     --unsupervised_exploration --exp_name q1_env1_random
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd
    #     --unsupervised_exploration --exp_name q1_env2_rnd
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0
    #     --unsupervised_exploration --exp_name q1_env2_random
    # ''')

    # #############################################
    # exp 2a
    # #############################################
    # all_cmds.append('''
    #     python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn
    #     --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql
    #     --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql_w_trans
    #     --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1
    #     --exploit_rew_shift 1.0 --exploit_rew_scale 100.0
    # ''')

    # # #############################################
    # # exp 2b
    # # #############################################
    for num_exploration_steps in [5000, 15000]:
        all_cmds.append(f'''
            python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd
            --num_exploration_steps={num_exploration_steps} --offline_exploitation --cql_alpha=0.1
            --unsupervised_exploration --exp_name q2_cql_numsteps_{num_exploration_steps}
        ''')

        all_cmds.append(f'''
            python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd
            --num_exploration_steps={num_exploration_steps} --offline_exploitation --cql_alpha=0.0
            --unsupervised_exploration --exp_name q2_dqn_numsteps_{num_exploration_steps}
        ''')

    # #############################################
    # exp 2c
    # #############################################
    for cql_alpha in [0.02, 0.5]:
        all_cmds.append(f'''
            python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd
            --unsupervised_exploration --offline_exploitation --cql_alpha={cql_alpha}
            --exp_name q2_alpha{cql_alpha}
        ''')

    # #############################################
    # exp 3
    # #############################################

    all_cmds.append('''
        python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd
        --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_medium_dqn
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd
        --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_medium_cql
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd
        --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_hard_dqn
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd
        --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_hard_cql
    ''')

    # # #############################################
    # # exp 4
    # # #############################################
    # for epsilon in [0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9]:
    #     all_cmds.append('''
    #         python cs285/scripts/run_hw3_dqn.py
    #         --env_name LunarLander-v3
    #         --exp_name q3_epsilon_{lander_epsilon}
    #         --lander_epsilon {lander_epsilon}
    #     '''.format(lander_epsilon=epsilon))

    for i, cmd in enumerate(all_cmds):
        last_time = time.time()

        train_w_parameters(cmd)

        print('---------------------------------------------')
        print(i, cmd)
        print('time used:', time.time()-last_time)
        print()

