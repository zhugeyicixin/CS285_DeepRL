import os
import shlex
import time
import argparse
import time
import itertools

from cs285.scripts.run_hw3_actor_critic import AC_Trainer


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iter', '-n', type=int, default=200)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--num_actor_updates_per_agent_update', type=int, default=1)

    parser.add_argument('--batch_size', '-b', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=400) #steps collected per eval iteration
    parser.add_argument('--train_batch_size', '-tb', type=int, default=1000) ##steps used per gradient step

    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    # parameters for parallelization
    # on the same core, use num_envs_per_core as batch size to generate actions using policy
    parser.add_argument('--num_envs_per_core', type=int, default=16)
    # steps collected per eval iteration
    parser.add_argument('--num_cores', type=int, default=1)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=10)

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

    # for policy gradient, we made a design decision
    # to force batch_size = train_batch_size
    # note that, to avoid confusion, you don't even have a train_batch_size argument anymore (above)
    params['train_batch_size'] = params['batch_size']

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

    ###################
    ### RUN TRAINING
    ###################

    trainer = AC_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    all_cmds = []
    os.chdir('../../')

    # #############################################
    # exp 4
    # #############################################
    # # env
    # all_cmds.append('''
    #     python run_hw3_actor_critic.py
    #     --env_name CartPole-v0
    #     -n 100 -b 1000
    #     --exp_name q4_ac_1_1
    #     -ntu 1 -ngsptu 1
    # ''')
    #
    # all_cmds.append('''
    #     python run_hw3_actor_critic.py
    #     --env_name CartPole-v0
    #     -n 100 -b 1000
    #     --exp_name q4_100_1
    #     -ntu 100 -ngsptu 1
    # ''')
    #
    # all_cmds.append('''
    #     python run_hw3_actor_critic.py
    #     --env_name CartPole-v0
    #     -n 100 -b 1000
    #     --exp_name q4_1_100
    #     -ntu 1 -ngsptu 100
    # ''')
    #
    # all_cmds.append('''
    #     python run_hw3_actor_critic.py
    #     --env_name CartPole-v0
    #     -n 100 -b 1000
    #     --exp_name q4_ac_10_10
    #     -ntu 10 -ngsptu 10
    # ''')

    # # #############################################
    # # exp 5
    # # #############################################
    # # env
    # ntu = 10
    # ngsptu = 10
    # all_cmds.append('''
    #     python run_hw3_actor_critic.py
    #     --env_name InvertedPendulum-v2
    #     --ep_len 1000 --discount 0.95
    #     -n 100 -l 2 -s 64
    #     -b 5000 -lr 0.01
    #     --exp_name q5_IP_{ntu}_{ngsptu}
    #     -ntu {ntu} -ngsptu {ngsptu}
    # '''.format(
    #     ntu=ntu,
    #     ngsptu=ngsptu
    # ))
    #
    # all_cmds.append('''
    #     python run_hw3_actor_critic.py
    #     --env_name HalfCheetah-v2
    #     --ep_len 150 --discount 0.90
    #     --scalar_log_freq 1
    #     -n 150 -l 2 -s 32
    #     -b 30000 -eb 1500 -lr 0.02
    #     --exp_name q5_HC_{ntu}_{ngsptu}
    #     -ntu {ntu} -ngsptu {ngsptu}
    # '''.format(
    #     ntu=ntu,
    #     ngsptu=ngsptu
    # ))

    # # #############################################
    # # video
    # # #############################################
    ntu = 10
    ngsptu = 10

    # all_cmds.append('''
    #     python run_hw3_actor_critic.py
    #     --env_name CartPole-v0
    #     -n 100 -b 1000
    #     --exp_name q4_ac_10_10
    #     -ntu 10 -ngsptu 10
    #     --video_log_freq 5 --num_envs_per_core 1 --num_cores 1
    # ''')

    all_cmds.append('''
        python run_hw3_actor_critic.py
        --env_name HalfCheetah-v2
        --ep_len 150 --discount 0.90
        --scalar_log_freq 1
        -n 150 -l 2 -s 32
        -b 30000 -eb 1500 -lr 0.02
        --exp_name q5_HC_{ntu}_{ngsptu}
        -ntu {ntu} -ngsptu {ngsptu}
        --video_log_freq 5 --num_envs_per_core 1 --num_cores 1
    '''.format(
        ntu=ntu,
        ngsptu=ngsptu
    ))

    for i, cmd in enumerate(all_cmds):
        last_time = time.time()

        train_w_parameters(cmd)

        print('---------------------------------------------')
        print(i, cmd)
        print('time used:', time.time()-last_time)
        print()

