"""
# @Author: JuQi
# @Time  : 2022/9/29 16:43
# @E-mail: 18672750887@163.com
"""

import numpy as np
from GFSP_Sampling.GFSP import GFSPSamplingSolver


class PCFRSolver(GFSPSamplingSolver):
    def __init__(self, config: dict):
        """
        GFSPSamplingSolver 的编程更优化版本
        :param config:
        """
        super().__init__(config)
        self.game.game_train_mode = 'PCFR'
        self.dynamic_now_prob = {}
        self.dynamic_opp_prob = {}
        self.dynamic_v = {}

    def walk_tree(self, his_feat, player_pi, pi_c):
        if self.game.game_train_mode == 'vanilla':
            return self.vanilla_walk_tree(his_feat, player_pi, pi_c)
        elif self.game.game_train_mode == 'PCFR':
            return self.PCFR_walk_tree(his_feat, player_pi, pi_c)
        return 0

    def PCFR_walk_tree(self, his_feat, player_pi, pi_c):
        self.node_touched += 1
        if self.game.player_num - np.count_nonzero(player_pi) == 2 or pi_c == 0:
            return np.zeros(self.game.player_num)

        now_player = self.game.get_now_player_from_his_feat(his_feat)
        if now_player == 'c':
            if self.sampling_mode == 'c_sampling':
                r = self.PCFR_walk_tree(his_feat + self.game.get_deterministic_chance_action(his_feat), player_pi, pi_c)
            else:
                r = np.zeros(self.game.player_num)
                now_prob = self.game.get_chance_prob(his_feat)
                now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)
                for a_i in range(len(now_action_list)):
                    tmp_r = self.PCFR_walk_tree(
                        self.game.get_next_his_feat(his_feat, now_action_list[a_i]),
                        player_pi,
                        pi_c * now_prob[a_i]
                    )
                    r += tmp_r
        else:
            now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)
            if len(now_action_list) == 0:
                if 0 in player_pi:
                    zero_index = np.where(player_pi == 0)[0][0]
                    tmp_reward = np.zeros(self.game.player_num)
                    tmp_reward[zero_index] = self.game.judge(his_feat)[zero_index]
                else:
                    tmp_reward = self.game.judge(his_feat)
                return tmp_reward * pi_c

            r = np.zeros(self.game.player_num)
            v = np.zeros(len(now_action_list))

            now_player_index = self.game.player_set.index(now_player)
            tmp_info = self.game.get_info_set(now_player, his_feat)
            now_prob = player_pi[now_player_index]

            if now_prob == 0:
                opp_prob = 1
                if self.sampling_mode == 'no_sampling':
                    if tmp_info in self.dynamic_opp_prob.keys():
                        self.dynamic_opp_prob[tmp_info] += now_prob
                    else:
                        self.dynamic_opp_prob[tmp_info] = now_prob

            else:
                if self.sampling_mode == 'no_sampling':
                    if tmp_info in self.dynamic_now_prob.keys():
                        self.dynamic_now_prob[tmp_info] += now_prob
                    else:
                        self.dynamic_now_prob[tmp_info] = now_prob

                if 0 in player_pi:
                    opp_prob = 0
                else:
                    opp_prob = 1
                    if self.sampling_mode == 'no_sampling':
                        if tmp_info in self.dynamic_opp_prob.keys():
                            self.dynamic_opp_prob[tmp_info] += now_prob
                        else:
                            self.dynamic_opp_prob[tmp_info] = now_prob

            if tmp_info not in self.game.imm_regret.keys():
                self.game.generate_new_info_set(tmp_info, now_player, len(now_action_list))

            if opp_prob == 0:
                a_i = self.game.now_policy[tmp_info]
                r = self.PCFR_walk_tree(his_feat + now_action_list[a_i], player_pi, pi_c)
                r[now_player_index] = 0
                if self.sampling_mode == 'c_sampling':
                    self.game.w_his_policy[tmp_info][self.game.now_policy[tmp_info]] += self.ave_weight

            elif now_prob == 0:
                for a_i in range(len(now_action_list)):
                    tmp_r = self.PCFR_walk_tree(his_feat + now_action_list[a_i], player_pi, pi_c)
                    v[a_i] = tmp_r[now_player_index]
                    if self.game.now_policy[tmp_info] == a_i:
                        r[now_player_index] = tmp_r[now_player_index]

                if self.sampling_mode == 'c_sampling':
                    self.game.imm_regret[tmp_info] += self.ave_weight * (v - v[self.game.now_policy[tmp_info]])
                    self.update_now_policy_P(tmp_info)
                else:
                    if tmp_info in self.dynamic_v.keys():
                        self.dynamic_v[tmp_info] += v
                    else:
                        self.dynamic_v[tmp_info] = v

            else:
                for a_i in range(len(now_action_list)):
                    if self.game.now_policy[tmp_info] == a_i:
                        prob = 1.0
                    else:
                        prob = 0.0

                    tmp_player_pi = np.ones(self.game.player_num)
                    tmp_player_pi[now_player_index] = prob

                    tmp_r = self.PCFR_walk_tree(his_feat + now_action_list[a_i], tmp_player_pi, pi_c)

                    v[a_i] = tmp_r[now_player_index]
                    r = r + tmp_r
                    if prob == 1:
                        pass
                    else:
                        r[now_player_index] = r[now_player_index] - tmp_r[now_player_index]

                if self.sampling_mode == 'c_sampling':
                    self.game.imm_regret[tmp_info] += self.ave_weight * (v - v[self.game.now_policy[tmp_info]])
                    self.game.w_his_policy[tmp_info][self.game.now_policy[tmp_info]] += self.ave_weight
                    self.update_now_policy_P(tmp_info)
                else:
                    if tmp_info in self.dynamic_v.keys():
                        self.dynamic_v[tmp_info] += v
                    else:
                        self.dynamic_v[tmp_info] = v
        return r

    def P_regret_matching_strategy(self, info):
        """
        在采样中的遗憾匹配
        :param info:
        :return:
        """

    def update_now_policy_P(self, info):
        tmp_pure_policy = np.random.randint(len(self.game.imm_regret[info]))
        self.game.now_policy[info] = np.argmax(self.game.imm_regret[info])
        if self.game.imm_regret[info][self.game.now_policy[info]] == self.game.imm_regret[info][tmp_pure_policy]:
            self.game.now_policy[info] = tmp_pure_policy

    def all_state_regret_matching_strategy(self):
        last_weight = 10000000000000000
        for info in self.dynamic_opp_prob.keys():
            q_max = np.max(self.game.imm_regret[info])
            q_gap = q_max - self.game.imm_regret[info]
            q_chasing = self.dynamic_v[info] - self.dynamic_v[info][self.game.now_policy[info]]
            if np.max(q_chasing) == 0:
                continue
            chasing_ge_zero = np.where(q_chasing > 0.0)
            tmp_weight = q_gap[chasing_ge_zero] // q_chasing[chasing_ge_zero] + 1
            last_weight = min(np.min(tmp_weight), last_weight)

            if last_weight == 1:
                break
        if last_weight == 10000000000000000:
            print(last_weight)

        for info in self.dynamic_now_prob.keys():
            self.game.w_his_policy[info][self.game.now_policy[info]] += last_weight
            if self.is_rm_plus:
                self.game.imm_regret[info][self.game.imm_regret[info] < 0] = 0.0

        for info in self.dynamic_opp_prob.keys():
            v = self.dynamic_v[info]
            self.game.imm_regret[info] += last_weight * (v - v[self.game.now_policy[info]])
            self.update_now_policy_P(info)

        self.dynamic_now_prob = {}
        self.dynamic_opp_prob = {}
        self.dynamic_v = {}
