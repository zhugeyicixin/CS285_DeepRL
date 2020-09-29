import os
import shlex
import time
import argparse
import time
import itertools

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

    # parameters for parallelization
    # on the same core, use num_envs_per_core as batch size to generate actions using policy
    parser.add_argument('--num_envs_per_core', type=int, default=16)
    # steps collected per eval iteration
    parser.add_argument('--num_cores', type=int, default=1)

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

    ##############################################
    # exp 1
    ##############################################
    # with video
    #         python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 300 -b 1000
    #         -dsa --exp_name q1_sb_no_rtg_dsa
    #         --eval_batch_size 1000
    #         --learning_rate 7.5e-4
    #         --video_log_freq 5 --num_envs_per_core 1 --num_cores 1

    # learning_rate = [2.5e-3, ]
    # discount = [1.0,  ]
    # n_layers = [2,]
    # size=[64,]
    # steps_per_iter=[1,]
    # # num_agent_train_steps_per_iter
    # for (lr, d, l, s, spi) in itertools.product(learning_rate, discount, n_layers, size, steps_per_iter):
    #     # env
    #     all_cmds.append('''
    #         python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000
    #         -rtg
    #         --exp_name q1_sb_rtg_na_lr{learning_rate}_d{discount}_l{n_layers}_s{size}_spi{steps_per_iter}
    #         --eval_batch_size 1000
    #         --learning_rate {learning_rate}
    #         --discount {discount}
    #         --n_layers {n_layers}
    #         --size {size}
    #         --num_agent_train_steps_per_iter {steps_per_iter}
    #         --num_envs_per_core 16 --num_cores 1
    #     '''.format(
    #         learning_rate=lr,
    #         discount=d,
    #         n_layers=l,
    #         size=s,
    #         steps_per_iter=spi
    #     ))

    # # env
    # all_cmds.append('''
    #     python cs285/scripts/run_hw2.py --env_name CartPole-v0
    #     -n 100 -b 1000
    #     -dsa --exp_name q1_sb_no_rtg_dsa
    #     --eval_batch_size 1000
    #     --learning_rate 2.5e-3
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw2.py --env_name CartPole-v0
    #     -n 100 -b 1000
    #     -rtg -dsa --exp_name q1_sb_rtg_dsa
    #     --eval_batch_size 1000
    #     --learning_rate 2.5e-3
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw2.py --env_name CartPole-v0
    #     -n 100 -b 1000
    #     -rtg --exp_name q1_sb_rtg_na
    #     --eval_batch_size 1000
    #     --learning_rate 2.5e-3
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw2.py --env_name CartPole-v0
    #     -n 100 -b 5000
    #     -dsa --exp_name q1_lb_no_rtg_dsa
    #     --eval_batch_size 1000
    #     --learning_rate 2.5e-3
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw2.py --env_name CartPole-v0
    #     -n 100 -b 5000
    #     -rtg -dsa --exp_name q1_lb_rtg_dsa
    #     --eval_batch_size 1000
    #     --learning_rate 2.5e-3
    # ''')
    #
    # all_cmds.append('''
    #     python cs285/scripts/run_hw2.py --env_name CartPole-v0
    #     -n 100 -b 5000
    #     -rtg --exp_name q1_lb_rtg_na
    #     --eval_batch_size 1000
    #     --learning_rate 2.5e-3
    # ''')

    # ###############################################
    # # exp 2
    # ###############################################
    # batch_size = [ 500, 1000, ]
    # learning_rate = [ 1e-2, 2.5e-2, 5e-2,  ]
    # discount = [0.9, ]
    # for (b, lr, d) in itertools.product(batch_size, learning_rate, discount):
    #     all_cmds.append('''
    #         python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2
    #         --ep_len 1000 --discount {discount} -n 100 -l 2 -s 64 -b {batch_size} -lr {learning_rate} -rtg
    #         --exp_name q2_b{batch_size}_r{learning_rate}
    #         --eval_batch_size 5000
    #     '''.format(
    #         batch_size=b,
    #         learning_rate=lr,
    #         discount=d,
    #     ))

    # all_cmds.append('''
    #     python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2
    #     --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 1000 -lr 0.05 -rtg
    #     --exp_name q2_b1000_r0.05
    #     --eval_batch_size 5000
    # ''')


    # ###############################################
    # # exp 3
    # ###############################################
    # all_cmds.append('''
    #     python cs285/scripts/run_hw2.py
    #     --env_name LunarLanderContinuous-v2 --ep_len 1000
    #     --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005
    #     --reward_to_go --nn_baseline --exp_name q3_b40000_r0.005
    #     --eval_batch_size 5000
    # ''')


    ###############################################
    # exp 4
    ###############################################
    # batch_size = [10000, 30000, 50000 ]
    # learning_rate = [5e-3, 0.01, 0.02]
    # for (b, lr) in itertools.product(batch_size, learning_rate):
    #     all_cmds.append('''
    #         python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150
    #         --discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {learning_rate} -rtg --nn_baseline
    #         --exp_name q4_search_b{batch_size}_lr{learning_rate}_rtg_nnbaseline
    #         --video_log_freq 5 --num_envs_per_core 1 --num_cores 1
    #     '''.format(
    #         batch_size=b,
    #         learning_rate=lr
    #     ))

    batch_size = [ 50000 ]
    learning_rate = [ 0.02]
    for (b, lr) in itertools.product(batch_size, learning_rate):
        # all_cmds.append('''
        #     python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150
        #     --discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {learning_rate}
        #     --exp_name q4_b{batch_size}_r{learning_rate}
        # '''.format(
        #     batch_size=b,
        #     learning_rate=lr
        # ))
        #
        # all_cmds.append('''
        #     python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150
        #     --discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {learning_rate} -rtg
        #     --exp_name q4_b{batch_size}_r{learning_rate}_rtg
        # '''.format(
        #     batch_size=b,
        #     learning_rate=lr
        # ))
        #
        # all_cmds.append('''
        #     python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150
        #     --discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {learning_rate} --nn_baseline
        #     --exp_name q4_b{batch_size}_r{learning_rate}_nnbaseline
        # '''.format(
        #     batch_size=b,
        #     learning_rate=lr
        # ))

        all_cmds.append('''
            python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150
            --discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {learning_rate} -rtg --nn_baseline
            --exp_name q4_b{batch_size}_r{learning_rate}_rtg_nnbaseline
        '''.format(
            batch_size=b,
            learning_rate=lr
        ))


    # ###############################################
    # # paralellization
    # ###############################################
    # batch_size = [10000, ]
    # learning_rate = [0.02,]
    # num_cores = [ 1, ]
    # num_envs_per_core = [16, ]
    # for (b, lr, nc, ne) in itertools.product(batch_size, learning_rate, num_cores, num_envs_per_core):
    #     all_cmds.append('''
    #         python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150
    #         --discount 0.95 -n 100 -l 2 -s 32 -b {batch_size} -lr {learning_rate} -rtg --nn_baseline
    #         --exp_name q4_search_b{batch_size}_lr{learning_rate}_rtg_nnbaseline_nc{num_cores}_ne{num_envs_per_core}
    #         --num_envs_per_core {num_envs_per_core} --num_cores {num_cores}
    #     '''.format(
    #         batch_size=b,
    #         learning_rate=lr,
    #         num_cores=nc,
    #         num_envs_per_core=ne,
    #     ))

    for i, cmd in enumerate(all_cmds):
        last_time = time.time()

        train_w_parameters(cmd)

        print('---------------------------------------------')
        print(i, cmd)
        print('time used:', time.time()-last_time)
        print()

