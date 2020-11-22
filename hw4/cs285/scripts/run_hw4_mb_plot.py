import json
import os

import matplotlib.pyplot as plt
import regex
import seaborn as sns
import pandas as pd
import numpy as np
from IPython.display import display


def plot_lr_q2():
    with open('../../data/good/q2_obstacles_singleiteration.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    data_2 = data_1.copy()
    data_2['AverageReturn'] = data_1['Train_AverageReturn']


    data_3 = data_1.copy()
    data_3['AverageReturn'] = data_1['Eval_AverageReturn']

    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
        'max_colwidth', -1
    ):
        display(data_1.head())
        display(data_2.head())
        display(data_3.head())


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)


    all_data = [data_2, data_3]
    all_colors = ['b', 'g']
    all_labels = ['Train_AverageReturn', 'Eval_AverageReturn', ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
            marker='*',
            markersize=30,
        )

    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    plt.xlim(-1, 1)
    # plt.xticks(
    #     np.arange(6)*100000,
    #     ['{:.0E}'.format(x) for x in np.arange(6)*100000]
    # )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.2, 0.8, 0.15, 0.1)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q2.png', dpi=300)
    plt.show()

def plot_lr_q3_1():
    with open('../../data/good/q3_obstacles.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    data_2 = data_1.copy()
    data_2['AverageReturn'] = data_1['Train_AverageReturn']


    data_3 = data_1.copy()
    data_3['AverageReturn'] = data_1['Eval_AverageReturn']

    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
        'max_colwidth', -1
    ):
        display(data_1.head())
        display(data_2.head())
        display(data_3.head())


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)


    all_data = [data_2, data_3]
    all_colors = ['b', 'g']
    all_labels = ['Train_AverageReturn', 'Eval_AverageReturn', ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
            marker='*',
            markersize=30,
        )

    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 1)
    # plt.xticks(
    #     np.arange(6)*100000,
    #     ['{:.0E}'.format(x) for x in np.arange(6)*100000]
    # )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.2, 0.8, 0.15, 0.1)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q3_1.png', dpi=300)
    plt.show()

def plot_lr_q3_2():
    with open('../../data/good/q3_reacher.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    data_2 = data_1.copy()
    data_2['AverageReturn'] = data_1['Train_AverageReturn']


    data_3 = data_1.copy()
    data_3['AverageReturn'] = data_1['Eval_AverageReturn']

    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
        'max_colwidth', -1
    ):
        display(data_1.head())
        display(data_2.head())
        display(data_3.head())


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)


    all_data = [data_2, data_3]
    all_colors = ['b', 'g']
    all_labels = ['Train_AverageReturn', 'Eval_AverageReturn', ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
            marker='*',
            markersize=30,
        )

    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 1)
    # plt.xticks(
    #     np.arange(6)*100000,
    #     ['{:.0E}'.format(x) for x in np.arange(6)*100000]
    # )
    plt.ylim(-2000, 0)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.2, 0.8, 0.15, 0.1)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q3_2.png', dpi=300)
    plt.show()

def plot_lr_q3_3():
    with open('../../data/good/q3_cheetah.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    data_2 = data_1.copy()
    data_2['AverageReturn'] = data_1['Train_AverageReturn']


    data_3 = data_1.copy()
    data_3['AverageReturn'] = data_1['Eval_AverageReturn']

    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
        'max_colwidth', -1
    ):
        display(data_1.head())
        display(data_2.head())
        display(data_3.head())


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)


    all_data = [data_2, data_3]
    all_colors = ['b', 'g']
    all_labels = ['Train_AverageReturn', 'Eval_AverageReturn', ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['AverageReturn'][:num_to_use],
            label=label,
            color=color,
            linewidth=5,
            marker='*',
            markersize=30,
        )

    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 1)
    # plt.xticks(
    #     np.arange(6)*100000,
    #     ['{:.0E}'.format(x) for x in np.arange(6)*100000]
    # )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.2, 0.8, 0.15, 0.1)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q3_3.png', dpi=300)
    plt.show()

def plot_lr_q4_1():
    with open('../../data/good/q4_reacher_horizon5.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q4_reacher_horizon15.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    with open('../../data/good/q4_reacher_horizon30.json', 'r') as fr:
        data_3 = json.load(fr)
    data_3 = pd.DataFrame(data_3)

    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
        'max_colwidth', -1
    ):
        display(data_1.head())
        display(data_2.head())
        display(data_3.head())


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, data_2, data_3]
    all_colors = sns.color_palette()
    all_labels = ['horizon=5', 'horizon=15', 'horizon=30']

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Train_AverageReturn'][:num_to_use],
            label='Train {}'.format(label),
            color=color,
            linewidth=5,
            marker='o',
            markersize=10,
            linestyle='-',
            fillstyle='none',
        )

        p1_2 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label='Eval {}'.format(label),
            color=color,
            linewidth=5,
            marker='o',
            markersize=10,
            linestyle='--',
        )

    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 1)
    # plt.xticks(
    #     np.arange(6)*100000,
    #     ['{:.0E}'.format(x) for x in np.arange(6)*100000]
    # )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.2, 0.8, 0.15, 0.1)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q4_1.png', dpi=300)
    plt.show()

def plot_lr_q4_2():
    with open('../../data/good/q4_reacher_numseq100.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q4_reacher_numseq1000.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
        'max_colwidth', -1
    ):
        display(data_1.head())
        display(data_2.head())


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, data_2, ]
    all_colors = sns.color_palette()
    all_labels = ['# of seq=100', '# of seq=1000', ]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Train_AverageReturn'][:num_to_use],
            label='Train {}'.format(label),
            color=color,
            linewidth=5,
            marker='o',
            markersize=10,
            linestyle='-',
            fillstyle='none',
        )

        p1_2 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label='Eval {}'.format(label),
            color=color,
            linewidth=5,
            marker='o',
            markersize=10,
            linestyle='--',
        )

    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 1)
    # plt.xticks(
    #     np.arange(6)*100000,
    #     ['{:.0E}'.format(x) for x in np.arange(6)*100000]
    # )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.2, 0.8, 0.15, 0.1)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q4_2.png', dpi=300)
    plt.show()

def plot_lr_q4_3():
    with open('../../data/good/q4_reacher_ensemble1.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q4_reacher_ensemble3.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    with open('../../data/good/q4_reacher_ensemble5.json', 'r') as fr:
        data_3 = json.load(fr)
    data_3 = pd.DataFrame(data_3)

    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.expand_frame_repr', False,
        'max_colwidth', -1
    ):
        display(data_1.head())
        display(data_2.head())
        display(data_3.head())


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, data_2, data_3]
    all_colors = sns.color_palette()
    all_labels = ['Ensemble size=1', 'Ensemble size=3', 'Ensemble size=5',]

    for i in range(len(all_data)):
        data = all_data[i]
        color = all_colors[i]
        label = all_labels[i]
        num_to_use = len(data['itr'])
        p1_1 = plt.plot(
            data['itr'][:num_to_use],
            data['Train_AverageReturn'][:num_to_use],
            label='Train {}'.format(label),
            color=color,
            linewidth=5,
            marker='o',
            markersize=10,
            linestyle='-',
            fillstyle='none',
        )

        p1_2 = plt.plot(
            data['itr'][:num_to_use],
            data['Eval_AverageReturn'][:num_to_use],
            label='Eval {}'.format(label),
            color=color,
            linewidth=5,
            marker='o',
            markersize=10,
            linestyle='--',
        )

    plt.xlabel('Number of iterations', size=36)
    plt.ylabel('Return', size=36)
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    # plt.xlim(-1, 1)
    # plt.xticks(
    #     np.arange(6)*100000,
    #     ['{:.0E}'.format(x) for x in np.arange(6)*100000]
    # )
    # plt.ylim(-500, 5500)

    # bbox_to_anchor (x, y, width, height)
    plt.legend(
        frameon=False,
        fontsize=32,
        loc='best',
        # bbox_to_anchor=(0.2, 0.8, 0.15, 0.1)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q4_3.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_lr_q2()

    # plot_lr_q3_1()
    # plot_lr_q3_2()
    # plot_lr_q3_3()
    #
    # plot_lr_q4_1()
    # plot_lr_q4_2()
    # plot_lr_q4_3()

