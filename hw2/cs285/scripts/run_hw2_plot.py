import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_training_step_per_iter():
    all_data = []
    all_nums = list(range(100, 1000, 100)) + list(range(1000, 10000, 1000))
    for num in all_nums:
        with open('../../data/bc_ant_{}.json'.format(num), 'r') as fr:
            d = json.load(fr)[0]
            d['num'] = num
            all_data.append(d)
    print('len(all_data)', len(all_data))

    df = pd.DataFrame(all_data)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)

    # p1 = plt.plot(
    #     df['num'],
    #     df['Eval_AverageReturn'],
    #     # lable='Ant-v2 BC',
    #     # linewidth=2,
    #     # size=5,
    #     # marker='o',
    #     # markersize=12,
    # )
    p2 = plt.errorbar(
        df['num'],
        df['Eval_AverageReturn'],
        yerr=df['Eval_StdReturn'],
        ecolor='red',
        elinewidth=2,
        capsize=5,
        capthick=2,
        marker='o',
        markersize=10,
        linewidth=3,
        label='Ant-v2 BC',
    )

    p3 = plt.plot(
        df['num'],
        df['Initial_DataCollection_AverageReturn'],
        label='Ant-v2 return of experts',
        color='g',
        linewidth=3,
    )

    plt.xlabel('Training steps per iteration', size=32)
    plt.ylabel('Return', size=32)
    ax.tick_params(axis='x', which='major', labelsize=24)
    ax.tick_params(axis='y', which='major', labelsize=24)
    plt.xlim(-500, 10500)
    plt.ylim(-800, 5800)

    plt.legend(frameon=False, fontsize=28)
    plt.show()

def plot_num_iterations():
    with open('../../data/bc_ant.json', 'r') as fr:
        bc_ant = json.load(fr)

    with open('../../data/dagger_ant.json', 'r') as fr:
        dagger_ant = json.load(fr)
    df_dagger_ant = pd.DataFrame(dagger_ant)

    with open('../../data/bc_humanoid.json', 'r') as fr:
        bc_humanoid = json.load(fr)

    with open('../../data/dagger_humanoid.json', 'r') as fr:
        dagger_humanoid = json.load(fr)
    df_dagger_humanoid = pd.DataFrame(dagger_humanoid)


    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111)

    sns.set(style="white", palette="muted", color_codes=True)
    paper_rc = {'lines.linewidth': 5}
    sns.set_context("paper", rc=paper_rc)


    num_to_use = 21
    p0 = plt.plot(
        df_dagger_ant['itr'][:num_to_use],
        df_dagger_ant['Initial_DataCollection_AverageReturn'][:num_to_use],
        label='Ant-v2 return of experts',
        color='c',
        linewidth=3,
    )

    p1 = plt.plot(
        df_dagger_ant['itr'][:num_to_use],
        [df_dagger_ant['Eval_AverageReturn'][0]]*num_to_use,
        label='Ant-v2 BC',
        color='m',
        linewidth=3,
    )

    p2 = plt.errorbar(
        df_dagger_ant['itr'][:num_to_use],
        df_dagger_ant['Eval_AverageReturn'][:num_to_use],
        yerr=df_dagger_ant['Eval_StdReturn'][:num_to_use],
        ecolor='blue',
        elinewidth=2,
        capsize=5,
        capthick=2,
        marker='o',
        markersize=10,
        color='blue',
        linewidth=3,
        label='Ant-v2 DAgger',
    )


    p0_2 = plt.plot(
        df_dagger_humanoid['itr'][:num_to_use],
        df_dagger_humanoid['Initial_DataCollection_AverageReturn'][:num_to_use],
        label='Humanoid-v2 return of experts',
        color='g',
        linewidth=3,
    )

    p1_2 = plt.plot(
        df_dagger_humanoid['itr'][:num_to_use],
        [df_dagger_humanoid['Eval_AverageReturn'][0]]*num_to_use,
        label='Humanoid-v2 BC',
        color='k',
        linewidth=3,
    )

    p2_2 = plt.errorbar(
        df_dagger_humanoid['itr'][:num_to_use],
        df_dagger_humanoid['Eval_AverageReturn'][:num_to_use],
        yerr=df_dagger_humanoid['Eval_StdReturn'][:num_to_use],
        ecolor='red',
        elinewidth=2,
        capsize=5,
        capthick=2,
        marker='s',
        markersize=10,
        color='red',
        linewidth=3,
        label='Humanoid-v2 DAgger',
    )

    plt.xlabel('Number of DAgger iterations', size=32)
    plt.ylabel('Return', size=32)
    ax.tick_params(axis='x', which='major', labelsize=28)
    ax.tick_params(axis='y', which='major', labelsize=28)
    plt.xlim(-1, 21)
    plt.xticks(range(21), range(1,22) )
    # plt.ylim(-500, 5500)

    plt.legend(
        frameon=False,
        fontsize=24,
        loc='best',
        bbox_to_anchor=(0.5, 0.6, 0.5, 0.4)
    )
    plt.show()


if __name__ == '__main__':
    # plot_training_step_per_iter()
    plot_num_iterations()