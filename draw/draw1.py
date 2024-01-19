"""
# @Author: JuQi
# @Time  : 2022/10/7 15:57
# @E-mail: 18672750887@163.com
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from draw.convergence_rate import plot_once


def plt_perfect_game_convergence_inline(game_name, logdir, is_x_log=True, is_y_log=True, y_num=3, x_num=0,
                                        log_interval_mode='node_touched'):
    file_list = os.listdir(logdir)
    file_list.sort()

    plt.ylabel('log10(Exploitability)')
    if log_interval_mode == 'node_touched':
        is_x_log = False
        is_y_log = True
        # plt.xlabel('log10(Node touched)')
        plt.xlabel('Node touched')
    elif log_interval_mode == 'train_time':
        is_x_log = False
        is_y_log = True
        plt.xlabel('Train Time')

    for i_file in range(len(file_list)):
        plot_once(
            logdir + '/' + file_list[i_file],
            i_file,
            file_list[i_file],
            is_x_log=is_x_log,
            is_y_log=is_y_log,
            y_num=y_num,
            x_num=x_num
        )

    plt.title(game_name)


if __name__ == '__main__':
    plt.figure(figsize=(32, 10), dpi=60)

    plt.subplot(1, 2, 1)
    plt_perfect_game_convergence_inline(
        '15_LeakyLeduc',
        '/home/root523/workspace/ft/new_gwpfefg/logGFSPSampling/important_15_LeakyLeduc',
        is_x_log=False,
        x_num=4,
        y_num=2,
        log_interval_mode='node_touched'
    )
    plt.subplot(1, 2, 2)
    plt_perfect_game_convergence_inline(
        '15_LeakyLeduc',
        '/home/root523/workspace/ft/new_gwpfefg/logGFSPSampling/important_15_LeakyLeduc',
        is_x_log=False,
        x_num=1,
        y_num=2,
        log_interval_mode='train_time'
    )

    plt.tight_layout()  # 自动调整子图、坐标轴和标题之间的间距，使得图像更紧凑，更美观。
    # plt.axis()
    plt.legend(edgecolor='red')  # 设置图例位置和颜色
    # plt.savefig(logdir + '/' + total_exp_name + '/pic.png')
    plt.show()
