"""
# @Author: JuQi
# @Time  : 2023/8/28 16:09
# @E-mail: 18672750887@163.com
"""
import numpy as np
import os
import time
import csv
import copy

from GAME_Sampling.GameZhaJinHuaNPlayer import ZhaJinHuaNPlayer

if __name__ == '__main__':
    file_list = os.listdir('league')
    print(file_list)
    league = []
    for i_policy_name in file_list:
        league.append(np.load('league' + '/' + i_policy_name, allow_pickle=True).item())

    league_len = len(league)

    game_config = {
        'game_name'      : 'Leduc',
        'prior_state_num': 4,
        'player_num'     : 3
    }

    game = ZhaJinHuaNPlayer(game_config)

    game.reset()
    tmp_reward = game.game_flow(
        {
            'player1': {},
            'player2': {},
            'player3': {},
        },
        is_show=True
    )
