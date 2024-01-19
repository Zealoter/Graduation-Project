"""
# @Author: JuQi
# @Time  : 2023/3/6 15:39
# @E-mail: 18672750887@163.com
"""
import copy

from GAME_Sampling.GameKuhn import Kuhn
import numpy as np


class KuhnNPlayer(Kuhn):
    def __init__(self, config):
        super().__init__(config)

        self.poker = [str(i + 1) for i in range(self.prior_state_num)]
        np.random.shuffle(self.poker)
        for i_player in range(self.player_num):
            self.pri_feat[self.player_set[i_player]] = str(self.poker[i_player])

    def get_now_player_from_his_feat(self, his_feat) -> str:
        if his_feat == '_':
            return 'c'
        else:
            tmp_his_feat = his_feat['his_feat']
            tmp_his_feat = tmp_his_feat.split('_')[-1]
            player_i = len(tmp_his_feat) % self.player_num
            return self.player_set[player_i]

    def judge(self, his_feat):
        tmp_h = his_feat['his_feat'].split('_')
        tmp_h = tmp_h[1:1 + self.player_num]
        poker_rank = np.array([int(i) for i in tmp_h])
        poker_rank = poker_rank * his_feat['alive_token']

        win_player = np.argmax(poker_rank)
        money = -copy.deepcopy(his_feat['player_pot'])
        total_pot = np.sum(his_feat['player_pot'])
        money[win_player] = total_pot - his_feat['player_pot'][win_player]
        money[0] = (money[0] + money[1]) / 2
        money[1] = money[0]
        return money

    def get_legal_action_list_from_his_feat(self, his_feat) -> list:
        if his_feat['his_feat'] == '_':
            start_c_action_list = []
            for poker1 in range(self.prior_state_num):
                for poker2 in range(self.prior_state_num):
                    if poker1 != poker2:
                        start_c_action_list.append(self.poker[poker1] + '_' + self.poker[poker2] + '_')
            return start_c_action_list

        tmp_feat = his_feat['his_feat'].split('_')[-1]
        if 'R' not in tmp_feat:
            if len(tmp_feat) >= self.player_num:
                return []
            else:
                return ['C', 'R']
        else:
            R_index = tmp_feat.index('R')
            if len(tmp_feat) - R_index == self.player_num:
                return []
            else:
                return ['F', 'C']

    def get_deterministic_chance_action(self, his_feat) -> str:
        tmp_act = copy.deepcopy(self.poker[:self.player_num])
        tmp_act = [str(i) for i in tmp_act]
        tmp_act = '_'.join(tmp_act)
        return tmp_act + '_'

    def reset(self):
        np.random.shuffle(self.poker)
        for i_player in range(self.player_num):
            self.pri_feat[self.player_set[i_player]] = str(self.poker[i_player])

    def get_pub_feat_from_his_feat(self, his_feat) -> str:
        if his_feat == '_':
            return his_feat
        else:
            pub_feat_list = his_feat['his_feat'].split('_')
            alive_code = [str(int(i)) for i in his_feat['alive_token']]
            alive_code = ''.join(alive_code)
            return '_' + alive_code + '_' + pub_feat_list[self.player_num + 1]

    def get_chance_prob(self, his_feat: str) -> np.ndarray:
        deal_num = 1
        for i in range(self.player_num):
            deal_num = deal_num * (self.prior_state_num - i)
        now_prob = np.ones(deal_num)
        now_prob = now_prob / np.sum(now_prob)
        return now_prob

    def get_info_set(self, player_id, his_feat):
        his_feat_list = his_feat['his_feat'].split('_')
        player_index = self.player_set.index(player_id)
        return his_feat_list[player_index + 1] + self.get_pub_feat_from_his_feat(his_feat)

    def get_next_his_feat(self, his_feat, now_action: str):
        # 大问题
        if his_feat == '_':
            his_feat = {
                'his_feat'   : '_' + now_action,
                'alive_token': np.ones(self.player_num),
                'player_pi'  : np.ones(self.player_num),
                'player_pot' : np.ones(self.player_num)
            }
            return his_feat
        else:
            next_his_feat = copy.deepcopy(his_feat)
            now_player = self.get_now_player_from_his_feat(his_feat)
            now_player_index = self.player_set.index(now_player)

            if now_action[-1] == 'X':
                pass
            elif now_action[-1] == 'F':
                next_his_feat['alive_token'][now_player_index] = 0
            elif now_action[-1] == 'R':
                next_his_feat['player_pot'][now_player_index] = 2
            elif now_action[-1] == 'C':
                if 'R' in next_his_feat['his_feat']:
                    next_his_feat['player_pot'][now_player_index] = 2

            next_his_feat['his_feat'] += now_action
            return next_his_feat
