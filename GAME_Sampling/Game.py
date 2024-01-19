"""
# @Author: JuQi
# @Time  : 2023/3/3 21:41
# @E-mail: 18672750887@163.com
"""

import numpy as np
import copy


class Game(object):
    def __init__(self, config: dict):
        self.game_name = config.get('game_name', '起个名字吧！')

        self.prior_state_num = config.get('prior_state_num', 3)
        self.player_num = config.get('player_num', 2)
        self.player_set = ['player' + str(i + 1) for i in range(self.player_num)]
        self.info_set_list = {}
        self.pri_feat = {}
        for p in self.player_set:
            self.info_set_list[p] = []
            self.pri_feat[p] = '_'
        self.terminal_list = []
        self.imm_regret = {}
        self.now_policy = {}
        self.now_prob = {}
        self.w_his_policy = {}

        self.itr = 0
        self.game_train_mode = 'vanilla'

    def generate_new_info_set(self, tmp_info_set, tmp_now_player, next_action_len):
        self.imm_regret[tmp_info_set] = np.zeros(next_action_len)
        # np.random.seed()
        if self.game_train_mode == 'PCFR':
            self.now_policy[tmp_info_set] = np.random.randint(next_action_len)
        else:
            self.now_policy[tmp_info_set] = np.random.random(next_action_len)
            self.now_policy[tmp_info_set] = self.now_policy[tmp_info_set] / np.sum(self.now_policy[tmp_info_set])

        self.w_his_policy[tmp_info_set] = np.zeros(next_action_len)
        self.info_set_list[tmp_now_player].append(tmp_info_set)
        self.now_prob[tmp_info_set] = 0

    def reset(self):
        """
        重置游戏
        :return:
        """
        pass

    def get_now_player_from_his_feat(self, his_feat: str) -> str:
        """
        根据历史流程，得到本轮玩家
        :param his_feat:
        :return: ['c','player1','player2',...]
        """
        pass

    def judge(self, his_feat) -> np.ndarray:
        """
        根据历史流程得到收益
        :param his_feat:
        :return:
        """
        pass

    def get_legal_action_list_from_his_feat(self, his_feat: str) -> list:
        """
        根据历史流程得到合法动作集
        :param his_feat:
        :return: 如果不是终端节点返回动作集，否则返回空集
        """
        pass

    def get_chance_prob(self, his_feat: str) -> np.ndarray:
        """
        根据历史流程机会节点的概率
        :param his_feat:
        :return: 返回各种可能的概率
        """
        pass

    def get_deterministic_chance_action(self, his_feat: str) -> str:
        """
        在MCCFR中，对于机会节点也可以采样，所以可以根据历史流程生成一个机会节点确定的动作。
        :param his_feat:
        :return: 确定性的动作
        """
        pass

    def is_pruning(self, his_feat) -> bool:
        if his_feat == '_':
            return False
        else:
            return len(np.where(his_feat['player_pi'] == 0)[0]) >= 2

    def get_sum_imm_regret(self):
        """
        得到历史总体遗憾值
        :return:
        """
        tmp_regret = copy.deepcopy(self.imm_regret)
        imm_regret_sum_per_player = np.zeros(len(self.player_set))

        for i_player in range(len(self.player_set)):
            for tmp_info in self.info_set_list[self.player_set[i_player]]:
                imm_regret_sum_per_player[i_player] += np.max(np.max(tmp_regret[tmp_info]), 0)  # 如果这是一个劣解，那么就要和0比较

        return imm_regret_sum_per_player

    def get_his_mean_policy(self) -> dict:
        """
        得到历史平均策略
        :return:
        """
        tmp_his_policy = copy.deepcopy(self.w_his_policy)
        for i_key in tmp_his_policy.keys():
            if np.sum(tmp_his_policy[i_key]) == 0:
                tmp_his_policy[i_key] = np.ones_like(tmp_his_policy[i_key]) / len(tmp_his_policy[i_key])
            else:
                tmp_his_policy[i_key] = tmp_his_policy[i_key] / np.sum(tmp_his_policy[i_key])
        return tmp_his_policy

    def get_next_his_feat(self, his_feat, now_action) -> str:
        """
        根据当前的历史流程和动作得到下一个历史流程
        :param his_feat:
        :param now_action:
        :return:
        """
        return his_feat + now_action

    def get_pub_feat_from_his_feat(self, his_feat: str) -> str:
        """
        通过历史流程得到现在玩家观测到的流程
        :param his_feat:
        :return:
        """
        return his_feat

    def get_info_set(self, player_id, his_feat):
        """
        得到当前的信息集
        :param player_id:
        :param his_feat:
        :return:
        """
        return self.pri_feat[player_id] + self.get_pub_feat_from_his_feat(his_feat)

    def game_flow(self, policy_list: dict, is_show=False):
        """
        模拟仿真对战
        :param is_show:
        :param policy_list:
        :return:
        """

        def game_sim(his_feat):
            now_player = self.get_now_player_from_his_feat(his_feat)
            if now_player == 'c':
                tmp_act = self.get_deterministic_chance_action(his_feat)
                # if is_show:
                #     print('机会节点', now_player, '回合')
                #     print('历史动作', his_feat)
                #     print('选择动作', tmp_act)
                return game_sim(self.get_next_his_feat(his_feat, tmp_act))
            else:
                now_info = self.get_info_set(now_player, his_feat)
                tmp_legal_actions = self.get_legal_action_list_from_his_feat(his_feat)
                if tmp_legal_actions:
                    if now_player in policy_list.keys():
                        if now_info in policy_list[now_player].keys():
                            tmp_act = np.random.choice(tmp_legal_actions, p=policy_list[now_player][now_info])
                        else:
                            tmp_act = np.random.choice(tmp_legal_actions)
                        if is_show:
                            print('AI玩家', now_player, '回合')
                            # print('历史动作', his_feat)
                            print('选择动作', tmp_act)
                        return game_sim(self.get_next_his_feat(his_feat, tmp_act))
                    else:
                        print('人类玩家回合')
                        print('当前信息集是：', now_info)
                        if self.game_name == 'ZhaJinHua' or self.game_name == 'ZhaJinHuaNPlayer':
                            if his_feat['is_shown'][self.player_set.index(now_player)] == 2:
                                print('手牌是：')
                                if now_player == 'player1':
                                    print(self.poker[:3])
                                elif now_player == 'player2':
                                    print(self.poker[3:6])
                                else:
                                    print(self.poker[6:9])

                        correct_input = False
                        while not correct_input:
                            try:
                                print('可以用的动作是：', tmp_legal_actions, '请输入(1,2,3...)')
                                human_act_index = input()
                                human_act = tmp_legal_actions[int(human_act_index) - 1]
                                correct_input = True
                            except:
                                print('输入错误，重新输入')

                        return game_sim(self.get_next_his_feat(his_feat, human_act))
                else:
                    result = self.judge(his_feat)
                    if is_show:
                        print('游戏结束：')
                        print('游戏全流程：', his_feat)
                        print('游戏分数：', result)
                        print('扑克牌：', self.poker)
                    return result

        return game_sim('_')

    def value_walk_tree(self, his_feat):
        now_action_list = self.get_legal_action_list_from_his_feat(his_feat)
        if len(now_action_list) == 0:
            tmp_reward = self.judge(his_feat)
            return tmp_reward
        r = np.zeros(len(self.player_set))
        now_player = self.get_now_player_from_his_feat(his_feat)
        if now_player == 'c':
            now_prob = self.get_chance_prob(his_feat)
            for a_i in range(len(now_action_list)):
                tmp_r = self.value_walk_tree(
                    self.get_next_his_feat(his_feat, now_action_list[a_i]),

                )
                r += tmp_r * now_prob[a_i]
        else:
            tmp_info = self.get_info_set(now_player, his_feat)
            if tmp_info not in self.now_policy.keys():
                tmp_policy = self.now_policy[tmp_info] = np.ones(len(now_action_list)) / len(now_action_list)
            else:
                tmp_policy = self.now_policy[tmp_info]
            for a_i in range(len(now_action_list)):
                tmp_r = self.value_walk_tree(
                    self.get_next_his_feat(his_feat, now_action_list[a_i])
                )
                r = r + tmp_r * tmp_policy[a_i]
        return r
