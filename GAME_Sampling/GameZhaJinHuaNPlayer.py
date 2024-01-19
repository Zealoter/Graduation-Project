import copy

from GAME_Sampling.GameZhaJinHua import ZhaJinHua
import numpy as np

def get_card_power(num1, num2, num3, color1, color2, color3):  # 获得牌力一般方法
    def get_num(tmp_num: str):
        if tmp_num.isdigit():
            return int(tmp_num)
        elif tmp_num == 'T':
            return 10
        elif tmp_num == 'J':
            return 11
        elif tmp_num == 'Q':
            return 12
        elif tmp_num == 'K':
            return 13
        else:
            return 14

    cards_list = [get_num(num1), get_num(num2), get_num(num3)]
    suites_list = [color1, color2, color3]  # 花色 映射到 1，2，3，4
    cards_list.sort()  # 从小到大排序
    result = 0
    abstraction_result = 0
    straight = 0
    if len(set(cards_list)) == 3:  # 3张不同牌
        if cards_list[0] + 2 == cards_list[2]:
            straight = 1
    # 6-三张 5-同花顺  4-同花  3-顺子  2-一对  1-高牌
    '''
        豹子 同花顺
        同花
        顺子（高牌）
        对子（高牌）
        高牌（高牌）
    '''
    if len(set(suites_list)) == 1:  # 同花
        if straight == 1:
            result = cards_list[2]
            result = 5 << 20 | result  # 同花顺
            abstraction_result = 4 << 20
        else:  # 同花
            for i in range(3):
                result = result | cards_list[i] << 4 * i
            abstraction_result = 4 << 20
            result = 4 << 20 | result
    elif straight == 1:
        result = cards_list[2]
        abstraction_result = 3 << 20 | result
        result = 3 << 20 | result  # 顺子
    elif len(set(cards_list)) == 1:  # 三条
        result = cards_list[0]
        result = 6 << 20 | result
        abstraction_result = 4 << 20
    else:
        if cards_list[0] == cards_list[1]:  # 一对
            abstraction_result = abstraction_result | cards_list[0] << 12
            abstraction_result = 2 << 20 | abstraction_result
            result = result | cards_list[0] << 12
            result = result | cards_list[2] << 8
            result = 2 << 20 | result

        elif cards_list[1] == cards_list[2]:
            abstraction_result = abstraction_result | cards_list[1] << 12
            abstraction_result = 2 << 20 | abstraction_result
            result = result | cards_list[1] << 12
            result = result | cards_list[0] << 8
            result = 2 << 20 | result

        elif cards_list[0] == cards_list[2]:
            abstraction_result = abstraction_result | cards_list[2] << 12
            abstraction_result = 2 << 20 | abstraction_result
            result = result | cards_list[2] << 12
            result = result | cards_list[1] << 8
            result = 2 << 20 | result

        else:
            for i in range(3):
                result = result | cards_list[i] << 4 * i
            abstraction_result = 1 << 20
            result = 1 << 20 | result
    return str(result), str(abstraction_result)

class ZhaJinHuaNPlayer(ZhaJinHua):
    def __init__(self, config):
        self.abstraction_pri_feat = {}
        super().__init__(config)
        self.action_cycle = 2 * self.player_num

    def get_card_strength(self):
        for i_player in range(self.player_num):
            self.pri_feat[self.player_set[i_player]], self.abstraction_pri_feat[self.player_set[i_player]] = \
                get_card_power(
                    self.poker[i_player * 3][1],
                    self.poker[i_player * 3 + 1][1],
                    self.poker[i_player * 3 + 2][1],
                    self.poker[i_player * 3][0],
                    self.poker[i_player * 3 + 1][0],
                    self.poker[i_player * 3 + 2][0],
                )

    def reset(self):
        np.random.shuffle(self.poker)
        self.get_card_strength()

    def get_next_his_feat(self, his_feat, now_action: str):
        # 大问题
        if his_feat == '_':
            his_feat = {
                'his_feat'      : '_' + now_action,
                'alive_token'   : np.ones(self.player_num),
                'player_pi'     : np.ones(self.player_num),
                'now_raise_size': 1,
                'is_shown'      : np.ones(self.player_num),
                'player_pot'    : np.ones(self.player_num)
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
            elif now_action[-1] == 'S':
                next_his_feat['is_shown'][now_player_index] = 2
            elif now_action[-1].isdigit():
                tmp_h = list(self.pri_feat.values())
                opp_player_index = int(now_action[-1]) - 1
                now_s = tmp_h[now_player_index]
                now_o = tmp_h[opp_player_index]
                next_his_feat['player_pot'][now_player_index] += \
                    (2 * next_his_feat['now_raise_size'] * next_his_feat['is_shown'][now_player_index])
                if now_s > now_o:
                    next_his_feat['alive_token'][opp_player_index] = 0
                else:
                    next_his_feat['alive_token'][now_player_index] = 0
            else:
                if now_action[-1] == 'r':
                    next_his_feat['now_raise_size'] = 2
                elif now_action[-1] == 'R':
                    next_his_feat['now_raise_size'] = 4
                elif now_action[-1] == 'C':
                    pass
                else:
                    pass
                next_his_feat['player_pot'][now_player_index] \
                    += (next_his_feat['now_raise_size'] * next_his_feat['is_shown'][now_player_index])
            next_his_feat['his_feat'] += now_action
            return next_his_feat

    def get_now_player_from_his_feat(self, his_feat) -> str:
        if his_feat == '_':
            return 'c'
        else:
            tmp_his_feat = his_feat['his_feat']
            i_player_num = (len(tmp_his_feat.split('_')[-1]) % self.action_cycle) // 2
            return self.player_set[i_player_num]

    def get_legal_action_list_from_his_feat(self, his_feat) -> list:
        tmp_his_feat = his_feat['his_feat']
        tmp_h = tmp_his_feat.split('_')[-1]
        now_player = self.get_now_player_from_his_feat(his_feat)
        now_player_idx = self.player_set.index(now_player)

        if len(tmp_h) >= self.action_cycle * 3:
            return []
        elif np.sum(his_feat['alive_token']) == 1:
            return []
        elif not his_feat['alive_token'][now_player_idx]:
            return ['XX']
        elif tmp_h.count('F') == self.player_num - 1:
            return []

        if 'R' in tmp_h:
            action_list = ['F', 'C']
        elif 'r' in tmp_h:
            action_list = ['F', 'C', 'R']
        else:
            action_list = ['F', 'C', 'r', 'R']
        if len(tmp_h) >= self.player_num * 2:
            alive_players = np.where(his_feat['alive_token'] == 1)[0]
            alive_players = list(alive_players)
            alive_players.remove(now_player_idx)
            for i_alive_player in alive_players:
                action_list.append(str(i_alive_player + 1))

        if 'S' in tmp_h[now_player_idx * 2::self.action_cycle]:
            if tmp_h[-1] == 'S':
                return action_list
            else:
                action_list = ['S' + i_act for i_act in action_list]
                return action_list
        else:
            action_list.remove('F')
            action_list = ['N' + i_act for i_act in action_list]
            return ['S'] + action_list

    def judge(self, his_feat):
        tmp_h = list(self.pri_feat.values())
        poker_rank = np.array([int(i) for i in tmp_h])
        poker_rank = poker_rank * his_feat['alive_token']
        win_player = np.argmax(poker_rank)
        money = -copy.deepcopy(his_feat['player_pot'])
        total_pot = np.sum(his_feat['player_pot'])
        money[win_player] = total_pot - his_feat['player_pot'][win_player]
        return money

    def get_deterministic_chance_action(self, his_feat) -> str:
        tmp_action = ''
        for i_player in self.player_set:
            tmp_action = tmp_action + self.abstraction_pri_feat[i_player] + '_'

        return tmp_action

    def get_info_set(self, player_id, his_feat):
        his_feat_list = his_feat['his_feat'].split('_')
        player_index = self.player_set.index(player_id)
        is_shown = his_feat_list[self.player_num + 1][player_index * 2::self.action_cycle]
        if 'S' in is_shown:
            self_feat = his_feat_list[player_index + 1]
        else:
            self_feat = ''

        return self_feat + self.get_pub_feat_from_his_feat(his_feat)

    def get_pub_feat_from_his_feat(self, his_feat) -> str:
        if his_feat == '_':
            return his_feat
        else:
            pub_feat_list = his_feat['his_feat'].split('_')
            alive_token = his_feat['alive_token']
            alive_token = [str(int(i)) for i in alive_token]
            alive_token = ''.join(alive_token)
            return '_' + alive_token + '_' + pub_feat_list[-1]
