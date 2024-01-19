"""
# @Author: JuQi
# @Time  : 2022/9/29 13:58
# @E-mail: 18672750887@163.com
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plot_color = [
    [
        (57 / 255, 81 / 255, 162 / 255),  # 深
        (202 / 255, 232 / 255, 242 / 255),  # 浅
        (114 / 255, 170 / 255, 207 / 255),  # 最浅

    ],
    [
        (168 / 255, 3 / 255, 38 / 255),
        (253 / 255, 185 / 255, 107 / 255),
        (236 / 255, 93 / 255, 59 / 255),
    ],
    [
        (0 / 255, 128 / 255, 51 / 255),
        (202 / 255, 222 / 255, 114 / 255),
        (226 / 255, 236 / 255, 179 / 255)
    ],
    [
        (128 / 255, 0 / 255, 128 / 255),
        (204 / 255, 153 / 255, 255 / 255),
        (128 / 255, 128 / 255, 128 / 255)
    ],
    [
        (255 / 255, 215 / 255, 0 / 255),  # 最深的黄色
        (255 / 255, 239 / 255, 213 / 255),  # 较浅的黄色
        (255 / 255, 250 / 255, 240 / 255),  # 最浅的黄色
    ],
    [
        (0 / 255, 0 / 255, 0 / 255),  # 最深的黑色
        (64 / 255, 64 / 255, 64 / 255),  # 较浅的黑色
        (128 / 255, 128 / 255, 128 / 255),  # 最浅的黑色
    ],
    [
        (255 / 255, 165 / 255, 0 / 255),  # 最深的橙色
        (255 / 255, 192 / 255, 128 / 255),  # 较浅的橙色
        (255 / 255, 224 / 255, 192 / 255),  # 最浅的橙色
    ],
]


def get_file_name_list(path: str) -> list:
    file_list = os.listdir(path)  # 获取path下面的所有文件，分类
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')
    csv_file = []
    for i_file in file_list:
        if i_file[-2:] == 'WS':
            continue
        csv_file.append(path + '/' + i_file + '/epsilon.csv')
    return csv_file


def get_result(file_path: str) -> np.ndarray:
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # 提取file_path这个文件里面的内容，转换为一个数组形式  ，表示里面的分隔符
    return data


def plot_once(path, num, ex_name, is_x_log=True, is_y_log=True, y_num=3, x_num=0):
    # 如果是时间x_num=1,如果是itr x_num=0
    csv_file_list = get_file_name_list(path)  # 获取子文件中所有的epsilon文件
    _10_num = int(0.1 * len(csv_file_list))
    tmp_data = get_result(csv_file_list[0])

    if is_x_log:
        tmp_x = np.log10(tmp_data[:, x_num])  # 训练次数    从tmp_data中取第一列的所有元素，并进行以10为底的log运算
    else:
        tmp_x = tmp_data[:, x_num]  # 否则直接讲第一列元素赋给tmp_x
    tmp_min_x = tmp_data.shape[0]
    tmp_y = tmp_data[:, y_num]  # 将第三列的值赋值给tmpy  即epsilon

    y_matrix = np.zeros((len(csv_file_list), tmp_min_x))  # 创建二维数组

    y_matrix[0, :] = tmp_data[:, y_num]  # 将tem_y全部元素赋值给二维数组的第一行
    for i in range(1, len(csv_file_list)):
        tmp_data = get_result(csv_file_list[i])
        now_min_x = tmp_data.shape[0]
        if now_min_x < tmp_min_x:
            tmp_min_x = now_min_x
            y_matrix = y_matrix[:, -tmp_min_x:]
            tmp_y = tmp_y[-tmp_min_x:]
            tmp_x = tmp_x[-tmp_min_x:]
        tmp_data = tmp_data[-tmp_min_x:, :]

        tmp_y += tmp_data[:, y_num]  # 对第y-num列的所有数据求和 分别对应相加
        y_matrix[i, :] = tmp_data[:, y_num]
        if is_y_log:
            plt.scatter(tmp_x, np.log10(tmp_data[:, y_num]), s=1, color=plot_color[num][1], alpha=0.7)
        else:
            plt.scatter(tmp_x, tmp_data[:, y_num], s=1, color=plot_color[num][1], alpha=0.3)
    y_matrix.sort(axis=0)  # 对y_matrix的第一列从小到大排序
    tmp_y = tmp_y / len(csv_file_list)  # 求平均
    if is_y_log:
        tmp_y = np.log10(tmp_y)
        y_matrix = np.log10(y_matrix)

    plt.plot(tmp_x, tmp_y, c=plot_color[num][0], lw=2, label=ex_name)
    plt.fill_between(tmp_x, y_matrix[_10_num, :], y_matrix[-_10_num - 1, :], color=plot_color[num][2], alpha=0.5)


def plot_hist(path, ex_name, y_num=2):
    csv_file_list = get_file_name_list(path)
    tmp_data = get_result(csv_file_list[0])

    tmp_x = [tmp_data[y_num]]

    for i in range(1, len(csv_file_list)):
        tmp_data = get_result(csv_file_list[i])
        tmp_x.append(tmp_data[y_num])
    sns.histplot(tmp_x, bins=20, color=plot_color[0][0], kde=True)
