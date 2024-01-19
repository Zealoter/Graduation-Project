"""
# @Author: JuQi
# @Time  : 2023/5/31 17:34
# @E-mail: 18672750887@163.com
"""
from GAME_Sampling.GameZhaJinHua import ZhaJinHua
from GAME_Sampling.GamePrincessAndMonster import PrincessAndMonster as PAM
from GAME_Sampling.GameLeduc import Leduc
import os
from GAME_Sampling.GameZhaJinHuaNPlayer import ZhaJinHuaNPlayer
from GAME_Sampling.GameKuhnNPlayer import KuhnNPlayer
import numpy as np


def play_more_games(games_time):
    human_player_reward = 0

    for i_game in range(games_time):
        print()
        print('****************************')
        print('第', i_game + 1, '局游戏：')
        print('****************************')
        game_config = {
            'game_name'      : 'ZhaJinHuaNPlayer',
            'game_info'      : 'ZhaJinHuaNPlayer',
            'prior_state_num': 7,
            'player_num'     : 3
        }

        game = ZhaJinHuaNPlayer(game_config)

        game.reset()
        if np.random.rand() < 1 / 3:
            tmp_reward = game.game_flow(
                {
                    'player2': AI_policy,
                    'player3': AI_policy,
                },
                is_show=True
            )
            human_player_reward += tmp_reward[0]
        elif np.random.rand() < 2 / 3:
            tmp_reward = game.game_flow(
                {
                    'player1': AI_policy,
                    'player3': AI_policy,
                },
                is_show=True
            )
            human_player_reward += tmp_reward[1]
        else:
            tmp_reward = game.game_flow(
                {
                    'player1': AI_policy,
                    'player2': AI_policy,
                },
                is_show=True
            )
            human_player_reward += tmp_reward[2]
        print('人类总收益：', human_player_reward)


if __name__ == '__main__':
    AI_policy = np.load(
        '/Users/juqi/Desktop/居奇综合/all_of_code/GWPFEFG/evaluate/AI/ZJH2108.npy',
        allow_pickle=True).item()
    # key = list(AI_policy.keys())
    # key.sort()
    # for i in key:
    #     print(i, AI_policy[i])

    play_more_games(30)
