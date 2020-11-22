import os
import shlex
import time
import argparse
import time
import itertools

from cs285.scripts.run_hw4_mb import MB_Trainer


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str) #reacher-cs285-v0, ant-cs285-v0, cheetah-cs285-v0, obstacles-cs285-v0
    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iter', '-n', type=int, default=20)

    parser.add_argument('--ensemble_size', '-e', type=int, default=3)
    parser.add_argument('--mpc_horizon', type=int, default=10)
    parser.add_argument('--mpc_num_action_sequences', type=int, default=1000)

    parser.add_argument('--add_sl_noise', '-noise', action='store_true')
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)
    parser.add_argument('--batch_size_initial', type=int, default=20000) #(random) steps collected on 1st iteration (put into replay buffer)
    parser.add_argument('--batch_size', '-b', type=int, default=8000) #steps collected per train iteration (put into replay buffer)
    parser.add_argument('--train_batch_size', '-tb', type=int, default=512) ##steps used per gradient step (used for training)
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=400) #steps collected per eval iteration

    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=250)

    # parameters for parallelization
    # on the same core, use num_envs_per_core as batch size to generate actions using policy
    parser.add_argument('--num_envs_per_core', type=int, default=16)
    # steps collected per eval iteration
    parser.add_argument('--num_cores', type=int, default=1)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=1) #-1 to disable
    parser.add_argument('--scalar_log_freq', type=int, default=1) #-1 to disable
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

    # HARDCODE EPISODE LENGTHS FOR THE ENVS USED IN THIS MB ASSIGNMENT
    if params['env_name']=='reacher-cs285-v0':
        params['ep_len']=200
    if params['env_name']=='cheetah-cs285-v0':
        params['ep_len']=500
    if params['env_name']=='obstacles-cs285-v0':
        params['ep_len']=100

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'hw4_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################

    trainer = MB_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    all_cmds = []
    os.chdir('../../')

    # # #############################################
    # # exp 1
    # # #############################################
    # # env
    # # Attention: the MPC policy only take in obs rather than batched obs_no
    # # Therefore, --num_envs_per_core 1 should be applied
    # all_cmds.append('''
    #     python cs285/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch1x32
    #     --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1
    #     --batch_size_initial 20000 --num_agent_train_steps_per_iter 500
    #     --n_layers 1 --size 32 --scalar_log_freq -1 --video_log_freq -1
    #     --num_envs_per_core 1
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw4_mb.py --exp_name q1_cheetah_n5_arch2x250
    #     --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1
    #     --batch_size_initial 20000 --num_agent_train_steps_per_iter 5
    #     --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1
    #     --num_envs_per_core 1
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch2x250
    #     --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1
    #     --batch_size_initial 20000 --num_agent_train_steps_per_iter 500
    #     --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1
    #     --num_envs_per_core 1
    # ''')


    # # #############################################
    # # exp 2
    # # #############################################
    # # env
    # # Attention: the MPC policy only take in obs rather than batched obs_no
    # # Therefore, --num_envs_per_core 1 should be applied
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw4_mb.py --exp_name
    #     q2_obstacles_singleiteration --env_name obstacles-cs285-v0
    #     --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1
    #     --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10
    #     --num_envs_per_core 1 --video_log_freq -1
    # ''')

    # #############################################
    # exp 3
    # #############################################
    # env
    # Attention: the MPC policy only take in obs rather than batched obs_no
    # Therefore, --num_envs_per_core 1 should be applied

    # all_cmds.append('''
    #     python cs285/scripts/run_hw4_mb.py --exp_name q3_obstacles
    #     --env_name obstacles-cs285-v0
    #     --add_sl_noise --num_agent_train_steps_per_iter 20
    #     --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10
    #     --n_iter 12
    #     --num_envs_per_core 1 --video_log_freq -1
    # ''')

    # all_cmds.append('''
    #     python cs285/scripts/run_hw4_mb.py --exp_name q3_reacher
    #     --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10
    #     --num_agent_train_steps_per_iter 1000 --batch_size_initial 5000
    #     --batch_size 5000 --n_iter 15
    #     --num_envs_per_core 1 --video_log_freq -1
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw4_mb.py --exp_name q3_cheetah
    #     --env_name cheetah-cs285-v0 --mpc_horizon 15 --add_sl_noise
    #     --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000
    #     --batch_size 5000 --n_iter 20
    #     --num_envs_per_core 1 --video_log_freq -1
    # ''')

    # #############################################
    # exp 4
    # #############################################
    # env
    # Attention: the MPC policy only take in obs rather than batched obs_no
    # Therefore, --num_envs_per_core 1 should be applied

    all_cmds.append('''
        python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_horizon5
        --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 5
        --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15
        --num_envs_per_core 1 --video_log_freq -1
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_horizon15
        --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 15
        --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15
        --num_envs_per_core 1 --video_log_freq -1
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_horizon30
        --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 30
        --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15
        --num_envs_per_core 1 --video_log_freq -1
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_numseq100
        --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10
        --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15
        --mpc_num_action_sequences 100
        --num_envs_per_core 1 --video_log_freq -1
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_numseq1000
        --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10
        --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15
        --mpc_num_action_sequences 1000
        --num_envs_per_core 1 --video_log_freq -1
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_ensemble1
        --env_name reacher-cs285-v0 --ensemble_size 1 --add_sl_noise
        --mpc_horizon 10 --num_agent_train_steps_per_iter 1000
        --batch_size 800 --n_iter 15
        --num_envs_per_core 1 --video_log_freq -1
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_ensemble3
        --env_name reacher-cs285-v0 --ensemble_size 3 --add_sl_noise
        --mpc_horizon 10 --num_agent_train_steps_per_iter 1000
        --batch_size 800 --n_iter 15
        --num_envs_per_core 1 --video_log_freq -1
    ''')

    all_cmds.append('''
        python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_ensemble5
        --env_name reacher-cs285-v0 --ensemble_size 5 --add_sl_noise
        --mpc_horizon 10 --num_agent_train_steps_per_iter 1000
        --batch_size 800 --n_iter 15
        --num_envs_per_core 1 --video_log_freq -1
    ''')

    for i, cmd in enumerate(all_cmds):
        last_time = time.time()

        train_w_parameters(cmd)

        print('---------------------------------------------')
        print(i, cmd)
        print('time used:', time.time()-last_time)
        print()

