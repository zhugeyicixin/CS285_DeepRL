import os
import shlex
import time
import argparse

from cs285.scripts.run_hw2 import PG_Trainer

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iter', '-n', type=int, default=200)

    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--nn_baseline', action='store_true')
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=400) #steps collected per eval iteration

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    parser.add_argument('--ep_len', type=int) #students shouldn't change this away from env's default
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=1)

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

    # convert args to dictionary
    params = vars(args)

    ## ensure compatibility with hw1 code
    params['train_batch_size'] = params['batch_size']

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################

    trainer = PG_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    all_cmds = []
    os.chdir('../../')

    # env
    all_cmds.append('''
        python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 
        -dsa --exp_name q1_sb_no_rtg_dsa
        --video_log_freq 5
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 
        -rtg -dsa --exp_name q1_sb_rtg_dsa
        --video_log_freq 5
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 
        -rtg --exp_name q1_sb_rtg_na
        --video_log_freq 5
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 
        -dsa --exp_name q1_lb_no_rtg_dsa
        --video_log_freq 5
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 
        -rtg -dsa --exp_name q1_lb_rtg_dsa
        --video_log_freq 5
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 
        -rtg --exp_name q1_lb_rtg_na
        --video_log_freq 5
    ''')



    # all_cmds.append('''
    #     python cs285/scripts/run_hw1.py
    #     --env_name Ant-v2 --exp_name bc_ant --n_iter 10
    #     --ep_len 1000 --eval_batch_size 5000
    #     --num_agent_train_steps_per_iter 2000
    #     --video_log_freq 5
    # ''')


    # all_nums = list(range(100, 1000, 100)) + list(range(1000, 10000, 1000))
    # for iter_num in all_nums:
    #     # bc_ant
    #     all_cmds.append('''
    #         python cs285/scripts/run_hw1.py
    #         --expert_policy_file cs285/policies/experts/Ant.pkl
    #         --env_name Ant-v2 --exp_name bc_ant_{iter_num} --n_iter 1
    #         --expert_data cs285/expert_data/expert_data_Ant-v2.pkl
    #         --ep_len 1000 --eval_batch_size 5000
    #         --num_agent_train_steps_per_iter {iter_num}
    #         --video_log_freq -1
    #         '''.format(iter_num=iter_num)
    #     )

    for cmd in all_cmds:
        train_w_parameters(cmd)

