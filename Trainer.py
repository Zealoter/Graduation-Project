"""
# @Author: JuQi
# @Time  : 2023/3/6 19:08
# @E-mail: 18672750887@163.com
"""
import copy
import time
import numpy as np
from GAME_Sampling.GameKuhn import Kuhn
from GAME_Sampling.GameLeduc import Leduc
from GAME_Sampling.GameKuhnNPlayer import KuhnNPlayer
from GAME_Sampling.GameZhaJinHua import ZhaJinHua
from GAME_Sampling.GameZhaJinHuaNPlayer import ZhaJinHuaNPlayer

from GFSP_Sampling.PCFR import PCFRSolver
from GFSP_Sampling.GFSP import GFSPSamplingSolver
from CONFIG import juqi_test_sampling_train_config
from CONFIG import ft_sampling_train_config

import draw.draw1

import draw.convergence_rate
from draw.convergence_rate import plot_once
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import gc


def train_sec(tmp_train_config):
    op_env = tmp_train_config.get('op_env', 'GFSP')
    if op_env == 'PCFR':
        tmp = PCFRSolver(tmp_train_config)
    elif op_env == 'GFSP':
        tmp = GFSPSamplingSolver(tmp_train_config)
    else:
        return
    tmp.train()
    del tmp
    gc.collect()
    return


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.6f}'.format}, suppress=True)

    logdir = 'logGFSPSampling'
    game_name = 'Leduc'
    is_show_policy = False
    prior_state_num = 3
    y_pot = 5
    z_len = 5

    game_config = {
        'game_name': game_name,
        'prior_state_num': prior_state_num,
        'y_pot': y_pot,
        'z_len': z_len,
        'player_num': 2,
        'depth': 4,
        'action_num': 4
    }

    if game_name == 'Leduc':
        game_class = Leduc(game_config)
    elif game_name == 'Kuhn':
        game_class = Kuhn(game_config)
    elif game_name == 'KuhnNPlayer':
        game_class = KuhnNPlayer(game_config)
    elif game_name == 'ZhaJinHua':
        game_class = ZhaJinHua(game_config)
    elif game_name == 'ZhaJinHuaNPlayer':
        game_class = ZhaJinHuaNPlayer(game_config)

    # train_mode = 'fix_itr'
    train_mode = 'fix_node_touched'
    # train_mode = 'fix_train_time'
    # log_interval_mode = 'itr'
    log_interval_mode = 'node_touched'
    # log_interval_mode = 'train_time'
    # log_mode = 'normal'
    log_mode = 'exponential'

    total_train_constraint = 1000000
    log_interval = 2
    nun_of_train_repetitions = 1
    n_jobs = 1  # 看看CPU有几个核再填（如果问题规模很小直接填1）

    total_exp_name = str(prior_state_num) + '_' + game_name + '_' + time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                                  time.localtime(time.time()))

    for key in juqi_test_sampling_train_config.keys():
        start = time.time()
        print(key)
        parallel_train_config_list = []
        for i_train_repetition in range(nun_of_train_repetitions):
            train_config = copy.deepcopy(juqi_test_sampling_train_config[key])

            train_config['game'] = copy.deepcopy(game_class)
            train_config['game_info'] = key
            train_config['train_mode'] = train_mode
            train_config['log_interval_mode'] = log_interval_mode
            train_config['log_mode'] = log_mode
            train_config['is_show_policy'] = is_show_policy

            train_config['total_exp_name'] = total_exp_name
            train_config['total_train_constraint'] = total_train_constraint
            train_config['log_interval'] = log_interval

            train_config['No.'] = i_train_repetition

            # train_config['fix_player'] = {
            #     'player1': def_policy,
            # }
            parallel_train_config_list.append(train_config)

        ans_list = Parallel(n_jobs=n_jobs)(
            delayed(train_sec)(i_train_config) for i_train_config in parallel_train_config_list
        )

        end = time.time()
        print(end - start)

    plt.figure(figsize=(32, 10), dpi=60)

    if game_name == 'KuhnNPot':
        fig_title = str(prior_state_num) + 'C' + str(y_pot) + 'P' + str(z_len) + 'L_Kuhn'
    else:
        fig_title = str(prior_state_num) + '_' + game_name

    if log_interval_mode == 'itr':
        plot_x_index = 0
    elif log_interval_mode == 'node_touched':
        plot_x_index = 4
    elif log_interval_mode == 'train_time':
        plot_x_index = 1
    else:
        print('plot_x_index指标有问题')
        plot_x_index = 0

    plt.subplot(1, 2, 1)
    draw.draw1.plt_perfect_game_convergence_inline(
        fig_title,
        logdir + '/' + total_exp_name,
        is_x_log=True,
        x_num=4,
        y_num=2,
        log_interval_mode='node_touched'
    )
    plt.subplot(1, 2, 2)
    draw.draw1.plt_perfect_game_convergence_inline(
        fig_title,
        logdir + '/' + total_exp_name,
        is_x_log=False,
        x_num=1,
        y_num=2,
        log_interval_mode='train_time'
    )

    plt.tight_layout()  # 自动调整子图、坐标轴和标题之间的间距，使得图像更紧凑，更美观。
    # plt.axis()
    plt.legend(edgecolor='red')  # 设置图例位置和颜色
    plt.savefig(logdir + '/' + total_exp_name + '/pic.png')
    # plt.show()
