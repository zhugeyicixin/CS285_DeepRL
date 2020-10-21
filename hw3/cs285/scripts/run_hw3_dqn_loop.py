import os
import shlex
import time
import argparse
import time
import itertools

from cs285.scripts.run_hw3_dqn import Q_Trainer


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_name',
        default='MsPacman-v0',
        choices=('PongNoFrameskip-v4', 'LunarLander-v3', 'MsPacman-v0')
    )

    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='todo')

    parser.add_argument('--eval_batch_size', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--double_q', action='store_true')
    parser.add_argument('--lander_epsilon', type=float, default=0.02)

    # parameters for parallelization
    # on the same core, use num_envs_per_core as batch size to generate actions using policy
    parser.add_argument('--num_envs_per_core', type=int, default=16)
    # steps collected per eval iteration
    parser.add_argument('--num_cores', type=int, default=1)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=int(1e4))
    parser.add_argument('--video_log_freq', type=int, default=-1)

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
    params['video_log_freq'] = -1 # This param is not used for DQN

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = 'hw3_' + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
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
    #     python cs285/scripts/run_hw3_dqn.py
    #     --env_name MsPacman-v0
    #     --exp_name q1
    # ''')

    # all_cmds.append('''
    #     python cs285/scripts/run_hw3_dqn.py
    #     --env_name LunarLander-v3
    #     --exp_name q1
    # ''')

    # # #############################################
    # # exp 2
    # # #############################################
    # all_cmds.append('''
    #     python cs285/scripts/run_hw3_dqn.py
    #     --env_name LunarLander-v3
    #     --exp_name q2_dqn_1
    #     --seed 1
    # ''')

    # all_cmds.append('''
    #     python cs285/scripts/run_hw3_dqn.py
    #     --env_name LunarLander-v3
    #     --exp_name q2_dqn_2
    #     --seed 2
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw3_dqn.py
    #     --env_name LunarLander-v3
    #     --exp_name q2_dqn_3
    #     --seed 3
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw3_dqn.py
    #     --env_name LunarLander-v3
    #     --exp_name q2_doubledqn_1
    #     --double_q
    #     --seed 1
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw3_dqn.py
    #     --env_name LunarLander-v3
    #     --exp_name q2_doubledqn_2
    #     --double_q
    #     --seed 2
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw3_dqn.py
    #     --env_name LunarLander-v3
    #     --exp_name q2_doubledqn_3
    #     --double_q
    #     --seed 3
    # ''')

    # #############################################
    # exp 3
    # #############################################
    for epsilon in [0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9]:
        all_cmds.append('''
            python cs285/scripts/run_hw3_dqn.py 
            --env_name LunarLander-v3 
            --exp_name q3_epsilon_{lander_epsilon}
            --lander_epsilon {lander_epsilon}
        '''.format(lander_epsilon=epsilon))

    for i, cmd in enumerate(all_cmds):
        last_time = time.time()

        train_w_parameters(cmd)

        print('---------------------------------------------')
        print(i, cmd)
        print('time used:', time.time()-last_time)
        print()

