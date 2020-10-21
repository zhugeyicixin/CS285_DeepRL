import json
import os

import matplotlib.pyplot as plt
import regex
import seaborn as sns
import pandas as pd
import numpy as np
from IPython.display import display

def plot_lr_q1():
    with open('../data/good/q1.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
        'max_colwidth', -1
    ):
        display(data_1.head())

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, ]
    all_colors = ['r', 'b', 'g']
    all_labels = ['DQN', 'lb_rtg_dsa', 'lb_rtg_na']

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Train_AverageReturn'][:num_to_use],
            label='{} average reward'.format(label),
            color='b',
            linewidth=5,
        )

        p1_2 = plt.plot(
            data['itr'][:num_to_use],
            data['Train_BestReturn'][:num_to_use],
            label='{} best reward'.format(label),
            color='g',
            linewidth=5,
        )


    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 21)
    plt.xticks(
        np.arange(6)*200000,
        ['{:.0E}'.format(x) for x in np.arange(6)*200000]
    )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.5, 0.2, 0.5, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../data/plots/lr_q1.png', dpi=300)
    plt.show()

def plot_lr_q2():
    with open('../data/good/q2_dqn_1.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../data/good/q2_dqn_2.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    with open('../data/good/q2_dqn_3.json', 'r') as fr:
        data_3 = json.load(fr)
    data_3 = pd.DataFrame(data_3)

    with open('../data/good/q2_doubledqn_1.json', 'r') as fr:
        data_4 = json.load(fr)
    data_4 = pd.DataFrame(data_4)

    with open('../data/good/q2_doubledqn_2.json', 'r') as fr:
        data_5 = json.load(fr)
    data_5 = pd.DataFrame(data_5)

    with open('../data/good/q2_doubledqn_3.json', 'r') as fr:
        data_6 = json.load(fr)
    data_6 = pd.DataFrame(data_6)

    data_1_avg = data_1.copy()
    data_1_avg['Train_AverageReturn'] = (
        data_1['Train_AverageReturn']
        + data_2['Train_AverageReturn']
        + data_3['Train_AverageReturn']
    )/3
    data_1_avg['Train_AverageReturn_min'] = np.min([
            data_1['Train_AverageReturn'],
            data_2['Train_AverageReturn'],
            data_3['Train_AverageReturn'],
        ],
        axis=0
    )
    data_1_avg['Train_AverageReturn_max'] = np.max([
            data_1['Train_AverageReturn'],
            data_2['Train_AverageReturn'],
            data_3['Train_AverageReturn'],
        ],
        axis=0
    )

    data_4_avg = data_4.copy()
    data_4_avg['Train_AverageReturn'] = (
        data_4['Train_AverageReturn']
        + data_5['Train_AverageReturn']
        + data_6['Train_AverageReturn']
    )/3
    data_4_avg['Train_AverageReturn_min'] = np.min([
            data_4['Train_AverageReturn'],
            data_5['Train_AverageReturn'],
            data_6['Train_AverageReturn'],
        ],
        axis=0
    )
    data_4_avg['Train_AverageReturn_max'] = np.max([
            data_4['Train_AverageReturn'],
            data_5['Train_AverageReturn'],
            data_6['Train_AverageReturn'],
        ],
        axis=0
    )

    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
        'max_colwidth', -1
    ):
        display(data_1.head())
        display(data_2.head())
        display(data_3.head())
        display(data_1_avg.head())


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)


    all_data = [data_1_avg, data_4_avg]
    all_colors = ['b', 'g']
    all_labels = ['DQN', 'DDQN', ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Train_AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
        )

        p1_2 = plt.fill_between(
            data['itr'][:num_to_use],
            data['Train_AverageReturn_min'][:num_to_use],
            data['Train_AverageReturn_max'][:num_to_use],
            color=color,
            alpha=0.2
        )


    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 21)
    plt.xticks(
        np.arange(6)*100000,
        ['{:.0E}'.format(x) for x in np.arange(6)*100000]
    )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.2, 0.8, 0.15, 0.1)
    )
    plt.tight_layout()
    plt.savefig('../data/plots/lr_q2.png', dpi=300)
    plt.show()


def plot_lr_q3():
    # goal
    all_data = []
    all_colors = sns.color_palette()
    all_labels = []

    # constant
    pattern_file = regex.compile('q3_(epsilon_[\.0-9]+).json')
    data_dir = '../data/good'
    files = os.listdir(data_dir)
    files = sorted(filter(lambda x: pattern_file.match(x), files))

    for f in files:
        with open(os.path.join(data_dir, f), 'r') as fr:
            data = json.load(fr)
        data = pd.DataFrame(data)
        all_data.append(data)

        all_labels.append(pattern_file.match(f).group(1))
    print('all_labels', all_labels)


    fig = plt.figure(figsize=(16, 12.5))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Train_AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
        )

        # p1_2 = plt.fill_between(
        #     data['itr'][:num_to_use],
        #     data['Eval_MinReturn'][:num_to_use],
        #     data['Eval_MaxReturn'][:num_to_use],
        #     color=color,
        #     alpha=0.2
        # )


    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 21)
    plt.xticks(
        np.arange(6)*100000,
        ['{:.0E}'.format(x) for x in np.arange(6)*100000]
    )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.5, 0.3, 0.15, 0.15)
    )
    plt.tight_layout()
    plt.savefig('../data/plots/lr_q3.png', dpi=300)
    plt.show()

def plot_lr_q4():
    with open('../data/good/q4_ac_1_1.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../data/good/q4_100_1.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    with open('../data/good/q4_1_100.json', 'r') as fr:
        data_3 = json.load(fr)
    data_3 = pd.DataFrame(data_3)

    with open('../data/good/q4_ac_10_10.json', 'r') as fr:
        data_4 = json.load(fr)
    data_4 = pd.DataFrame(data_4)

    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
        'max_colwidth', -1
    ):
        display(data_1.head())

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, data_2, data_3, data_4,]
    all_colors = sns.color_palette()
    all_labels = [
        'ntu=1, ngsptu=1',
        'ntu=100, ngsptu=1',
        'ntu=1, ngsptu=100',
        'ntu=10, ngsptu=10',
    ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label='{}'.format(label),
            color=color,
            linewidth=5,
        )

    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 21)
    # plt.xticks(
    #     np.arange(6)*200000,
    #     ['{:.0E}'.format(x) for x in np.arange(6)*200000]
    # )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.5, 0.2, 0.5, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../data/plots/lr_q4.png', dpi=300)
    plt.show()

def plot_lr_q5_IP():
    with open('../data/good/q5_IP_10_10.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
        'max_colwidth', -1
    ):
        display(data_1.head())

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, ]
    all_colors = sns.color_palette()
    all_labels = [
        'InvertedPendulum-v2',
    ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label='{}'.format(label),
            color=color,
            linewidth=5,
        )

    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 21)
    # plt.xticks(
    #     np.arange(6)*200000,
    #     ['{:.0E}'.format(x) for x in np.arange(6)*200000]
    # )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.5, 0.2, 0.5, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../data/plots/lr_q5_IP.png', dpi=300)
    plt.show()


def plot_lr_q5_HC():
    with open('../data/good/q5_HC_10_10.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)


    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
        'max_colwidth', -1
    ):
        display(data_2.head())

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)


    all_data = [ data_2,]
    all_colors = sns.color_palette()
    all_labels = [
        'HalfCheetah-v2',
    ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label='{}'.format(label),
            color=color,
            linewidth=5,
        )

    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 21)
    # plt.xticks(
    #     np.arange(6)*200000,
    #     ['{:.0E}'.format(x) for x in np.arange(6)*200000]
    # )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.5, 0.2, 0.5, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../data/plots/lr_q5_HC.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    # plot_lr_q1()
    #
    # plot_lr_q2()
    #
    # plot_lr_q3()
    #
    # plot_lr_q4()

    plot_lr_q5_IP()
    plot_lr_q5_HC()
