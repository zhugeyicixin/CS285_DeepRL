import json
import os

import matplotlib.pyplot as plt
import regex
import seaborn as sns
import pandas as pd


def plot_lr_q1_lb():
    with open('../../data/good/q1_lb_no_rtg_dsa.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q1_lb_rtg_dsa.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    with open('../../data/good/q1_lb_rtg_na.json', 'r') as fr:
        data_3 = json.load(fr)
    data_3 = pd.DataFrame(data_3)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, data_2, data_3]
    all_colors = ['r', 'b', 'g']
    all_labels = ['lb_no_rtg_dsa', 'lb_rtg_dsa', 'lb_rtg_na']

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
    plt.savefig('../../data/plots/lr_q1_lb.png', dpi=300)
    plt.show()

def plot_lr_q1_sb():
    with open('../../data/good/q1_sb_no_rtg_dsa.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q1_sb_rtg_dsa.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    with open('../../data/good/q1_sb_rtg_na.json', 'r') as fr:
        data_3 = json.load(fr)
    data_3 = pd.DataFrame(data_3)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, data_2, data_3]
    all_colors = ['r', 'b', 'g']
    all_labels = ['sb_no_rtg_dsa', 'sb_rtg_dsa', 'sb_rtg_na']

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
        bbox_to_anchor=(0.3, 0.1, 0.3, 0.3)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q1_sb.png', dpi=300)
    plt.show()

def plot_lr_q2():
    with open('../../data/good/q2_b1000_r0.05.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, ]
    all_colors = ['g']
    all_labels = ['InvertedPendulum']

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
        # bbox_to_anchor=(0.2, 0.8, 0.15, 0.1)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q2.png', dpi=300)
    plt.show()

def plot_lr_q3():
    with open('../../data/good/q3_b40000_r0.005.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, ]
    all_colors = ['g']
    all_labels = ['LunarLander']

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
        bbox_to_anchor=(0.5, 0.3, 0.15, 0.15)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q3.png', dpi=300)
    plt.show()


def plot_lr_q4_para_grid():
    # goal
    all_data = []
    all_colors = sns.color_palette()
    all_labels = []

    # constant
    pattern_file = regex.compile('q4_search_(b[0-9]+_lr[\.0-9]+)_rtg_nnbaseline.json')
    data_dir = '../../data/good'
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
        # bbox_to_anchor=(0.5, 0.3, 0.15, 0.15)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q4_para_grid.png', dpi=300)
    plt.show()


def plot_lr_q4_four():
    with open('../../data/good/q4_b50000_r0.02.json', 'r') as fr:
        data_1 = json.load(fr)
    data_1 = pd.DataFrame(data_1)

    with open('../../data/good/q4_b50000_r0.02_rtg.json', 'r') as fr:
        data_2 = json.load(fr)
    data_2 = pd.DataFrame(data_2)

    with open('../../data/good/q4_b50000_r0.02_nnbaseline.json', 'r') as fr:
        data_3 = json.load(fr)
    data_3 = pd.DataFrame(data_3)

    with open('../../data/good/q4_b50000_r0.02_rtg_nnbaseline.json', 'r') as fr:
        data_4 = json.load(fr)
    data_4 = pd.DataFrame(data_4)

    fig = plt.figure(figsize=(12.5, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    # paper_rc = {'lines.linewidth': 5}
    # sns.set_context("paper", rc=paper_rc)


    all_data = [data_1, data_2, data_3, data_4]
    all_colors = ['k', 'r', 'b', 'g']
    all_labels = [
        'w/o rtg & baseline',
        'w/ only rtg',
        'w/ only baseline',
        'w/ rtg & baseline'
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
        bbox_to_anchor=(0.45, 0.28, 0.15, 0.15)
    )
    plt.tight_layout()
    plt.savefig('../../data/plots/lr_q4_four.png', dpi=300)
    plt.show()



if __name__ == '__main__':
    plot_lr_q1_lb()
    plot_lr_q1_sb()

    plot_lr_q2()
    plot_lr_q3()

    plot_lr_q4_para_grid()
    plot_lr_q4_four()