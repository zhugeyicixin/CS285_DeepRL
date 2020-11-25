import json
import os

import matplotlib.pyplot as plt
import regex
import seaborn as sns
import pandas as pd
import numpy as np
from IPython.display import display


def plot_lr_q1_1():
    with open('../../data/good/q1_env1_rnd.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q1_env1_random.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, data_2, ]
    all_colors = ['r', 'b', 'g']
    all_labels = ['RND', 'random exploration', ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
        )

        p1_2 = plt.fill_between(
            data['itr'][:num_to_use],
            data['Eval_MinReturn'][:num_to_use],
            data['Eval_MaxReturn'][:num_to_use],
            color=color,
            alpha=0.2
        )


    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Eval Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 21)
    # plt.xticks(range(21), range(1,22) )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        bbox_to_anchor=(0.5, 0.2, 0.5, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q1_1.png', dpi=300)
    plt.show()


def plot_lr_q1_2():
    with open('../../data/good/q1_env2_rnd.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q1_env2_random.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, data_2, ]
    all_colors = ['r', 'b', 'g']
    all_labels = ['RND', 'random\nexploration', ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
        )

        p1_2 = plt.fill_between(
            data['itr'][:num_to_use],
            data['Eval_MinReturn'][:num_to_use],
            data['Eval_MaxReturn'][:num_to_use],
            color=color,
            alpha=0.2
        )


    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Eval Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 21)
    # plt.xticks(range(21), range(1,22) )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        bbox_to_anchor=(0.5, 0.2, 0.5, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q1_2.png', dpi=300)
    plt.show()

def plot_lr_q2_1():
    with open('../../data/good/q2_dqn.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q2_cql.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    with open('../../data/good/q2_cql_w_trans.json', 'r') as fr:
        data_3 = json.load(fr)
    data_3 = pd.DataFrame(data_3)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, data_2, data_3, ]
    all_colors = sns.color_palette()
    all_labels = [
        'DQN',
        'CQL',
        'CQL w/\ntransformed\nreward',
    ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
        )

        p1_2 = plt.fill_between(
            data['itr'][:num_to_use],
            data['Eval_MinReturn'][:num_to_use],
            data['Eval_MaxReturn'][:num_to_use],
            color=color,
            alpha=0.2
        )

    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Eval Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 21)
    # plt.xticks(range(21), range(1,22) )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        bbox_to_anchor=(0.6, 0.2, 0.3, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q2_1.png', dpi=300)
    plt.show()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p2_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Exploitation Data q-values'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
        )


    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Exploitation Q-values', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    plt.xlim(0, 50000)
    # plt.xticks(range(21), range(1,22) )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        bbox_to_anchor=(0.6, 0.2, 0.3, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/Q_q2_1.png', dpi=300)
    plt.show()


def plot_lr_q2_2():
    with open('../../data/good/q2_dqn_numsteps_5000.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q2_cql_numsteps_5000.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    with open('../../data/good/q2_dqn_numsteps_15000.json', 'r') as fr:
        data_3 = json.load(fr)
    data_3 = pd.DataFrame(data_3)

    with open('../../data/good/q2_cql_numsteps_15000.json', 'r') as fr:
        data_4 = json.load(fr)
    data_4 = pd.DataFrame(data_4)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, data_2, data_3, data_4 ]
    all_colors = sns.color_palette()
    all_labels = [
        'DQN expl. steps 5000',
        'CQL expl. steps 5000',
        'DQN expl. steps 15000',
        'CQL expl. steps 15000',
    ]

    for i in range(len(all_data)):
        data = all_data[i]
        print(all_labels[i])
        print(max(data['Eval_AverageReturn']))

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
        )

        p1_2 = plt.fill_between(
            data['itr'][:num_to_use],
            data['Eval_MinReturn'][:num_to_use],
            data['Eval_MaxReturn'][:num_to_use],
            color=color,
            alpha=0.2
        )


    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Eval Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    plt.xlim(0, 50000)
    # plt.xticks(range(21), range(1,22) )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=28,
        loc='best',
        bbox_to_anchor=(0.45, 0.30, 0.3, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q2_2.png', dpi=300)
    plt.show()

def plot_lr_q2_3():
    with open('../../data/good/q2_alpha0.02.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q2_alpha0.5.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    with open('../../data/good/q2_cql.json', 'r') as fr:
        data_3 = json.load(fr)
    data_3 = pd.DataFrame(data_3)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, data_3, data_2,  ]
    all_colors = sns.color_palette()
    all_labels = [
        'Alpha 0.02',
        'Alpha 0.1',
        'Alpha 0.5',
    ]

    for i in range(len(all_data)):
        data = all_data[i]
        print(all_labels[i])
        print(max(data['Eval_AverageReturn']))

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
        )

        p1_2 = plt.fill_between(
            data['itr'][:num_to_use],
            data['Eval_MinReturn'][:num_to_use],
            data['Eval_MaxReturn'][:num_to_use],
            color=color,
            alpha=0.2
        )


    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Eval Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 21)
    # plt.xticks(range(21), range(1,22) )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=28,
        loc='best',
        bbox_to_anchor=(0.07, 0.7, 0.3, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q2_3.png', dpi=300)
    plt.show()

def plot_lr_q3_1():
    with open('../../data/good/q3_medium_cql.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q3_medium_dqn.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)

    all_data = [data_1, data_2,  ]
    all_colors = sns.color_palette()
    all_labels = [
        'Medium CQL',
        'Medium DQN',
    ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
        )

        p1_2 = plt.fill_between(
            data['itr'][:num_to_use],
            data['Eval_MinReturn'][:num_to_use],
            data['Eval_MaxReturn'][:num_to_use],
            color=color,
            alpha=0.2
        )

    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Eval Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 21)
    # plt.xticks(range(21), range(1,22) )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        bbox_to_anchor=(0.6, 0.2, 0.3, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q3_1.png', dpi=300)
    plt.show()

def plot_lr_q3_2():
    with open('../../data/good/q3_hard_cql.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q3_hard_dqn.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, data_2,  ]
    all_colors = sns.color_palette()
    all_labels = [
        'Hard CQL',
        'HARD DQN',
    ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
        )

        p1_2 = plt.fill_between(
            data['itr'][:num_to_use],
            data['Eval_MinReturn'][:num_to_use],
            data['Eval_MaxReturn'][:num_to_use],
            color=color,
            alpha=0.2
        )


    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Eval Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 21)
    # plt.xticks(range(21), range(1,22) )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.6, 0.2, 0.3, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q3_2.png', dpi=300)
    plt.show()


def plot_lr_q3_3():
    with open('../../data/good/q3_medium_cql.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q3_medium_dqn.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    with open('../../data/good/q2_cql_numsteps_15000.json', 'r') as fr:
        data_3 = json.load(fr)
    data_3 = pd.DataFrame(data_3)

    with open('../../data/good/q2_dqn_numsteps_15000.json', 'r') as fr:
        data_4 = json.load(fr)
    data_4 = pd.DataFrame(data_4)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)

    all_data = [data_1, data_2, data_3, data_4 ]
    all_colors = sns.color_palette()
    all_labels = [
        'Supervised CQL',
        'Supervised DQN',
        'Unsupervised CQL',
        'Unsupervised DQN',
    ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
        )

        p1_2 = plt.fill_between(
            data['itr'][:num_to_use],
            data['Eval_MinReturn'][:num_to_use],
            data['Eval_MaxReturn'][:num_to_use],
            color=color,
            alpha=0.2
        )

    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Eval Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 21)
    # plt.xticks(range(21), range(1,22) )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=28,
        loc='best',
        bbox_to_anchor=(0.55, 0.2, 0.3, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q3_3.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # plot_lr_q1_1()
    # plot_lr_q1_2()
    #
    # plot_lr_q2_1()
    # plot_lr_q2_2()
    # plot_lr_q2_3()
    #
    # plot_lr_q3_1()
    # plot_lr_q3_2()
    plot_lr_q3_3()

