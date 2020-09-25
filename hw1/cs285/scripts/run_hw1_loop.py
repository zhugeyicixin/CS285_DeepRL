import os
import shlex
import time
import argparse

from cs285.scripts.run_hw1 import BC_Trainer

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help='choices: Ant-v2, Humanoid-v2, Walker2d-v2, HalfCheetah-v2, Hopper-v2', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=1000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
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

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if args.do_dagger:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q2_'
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q1_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)


    ###################
    ### RUN TRAINING
    ###################

    trainer = BC_Trainer(params)
    trainer.run_training_loop()

if __name__ == "__main__":
    all_cmds = []
    os.chdir('../../')

    # bc_ant
    all_cmds.append('''
        python cs285/scripts/run_hw1.py
        --expert_policy_file cs285/policies/experts/Ant.pkl
        --env_name Ant-v2 --exp_name bc_ant --n_iter 1
        --expert_data cs285/expert_data/expert_data_Ant-v2.pkl
        --ep_len 1000 --eval_batch_size 5000
        --num_agent_train_steps_per_iter 2000
        --video_log_freq -1
    ''')

    # bc_humanoid
    all_cmds.append('''
        python cs285/scripts/run_hw1.py
        --expert_policy_file cs285/policies/experts/Humanoid.pkl
        --env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1
        --expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl
        --ep_len 1000 --eval_batch_size 5000
        --num_agent_train_steps_per_iter 2000
        --video_log_freq -1
    ''')

    # dagger_ant
    all_cmds.append('''
        python cs285/scripts/run_hw1.py 
        --expert_policy_file cs285/policies/experts/Ant.pkl 
        --env_name Ant-v2 --exp_name dagger_ant --n_iter 21 
        --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl 
        --ep_len 1000 --eval_batch_size 5000 
        --num_agent_train_steps_per_iter 2000 
        --batch_size 5000 
        --video_log_freq -1
    ''')

    # dagger_humanoid
    all_cmds.append('''
        python cs285/scripts/run_hw1.py 
        --expert_policy_file cs285/policies/experts/Humanoid.pkl 
        --env_name Humanoid-v2 --exp_name dagger_humanoid --n_iter 21 
        --do_dagger --expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl 
        --ep_len 1000 --eval_batch_size 5000 
        --num_agent_train_steps_per_iter 2000 
        --batch_size 5000 
        --video_log_freq -1
    ''')

    all_nums = list(range(100, 1000, 100)) + list(range(1000, 10000, 1000))
    for iter_num in all_nums:
        # bc_ant
        all_cmds.append('''
            python cs285/scripts/run_hw1.py
            --expert_policy_file cs285/policies/experts/Ant.pkl
            --env_name Ant-v2 --exp_name bc_ant_{iter_num} --n_iter 1
            --expert_data cs285/expert_data/expert_data_Ant-v2.pkl
            --ep_len 1000 --eval_batch_size 5000
            --num_agent_train_steps_per_iter {iter_num}
            --video_log_freq -1
            '''.format(iter_num=iter_num)
        )

    for cmd in all_cmds:
        train_w_parameters(cmd)

