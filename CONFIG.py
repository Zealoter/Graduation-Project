"""
# @Author: JuQi
# @Time  : 2023/3/10 19:33
# @E-mail: 18672750887@163.com
"""

"""
    'game'：是游戏的环境
    'rm_mode'：可以选
        vanilla：代表原始CFR，
        eta：代表改RM时的权重，如果选了eta模式就一定要设置rm_eta值
        br：就是最佳响应
    'is_rm_plus'：代表是否采用CFR+
    'sampling_mode'：
        如果是'c_sampling':全采样
        如果是'no_sampling':不采样
    'ave_mode' :
        'vanilla': 自然平均
        'log': T的对数
        'liner': T的线性

    'lr'：待完善
    'log_mode':
        'exponential':
        'normal':
"""
ft_sampling_train_config = {
    'PCFR'                   : {
        'game'         : None,
        'rm_mode'      : 'br',  # CFR
        'rm_eta'       : 1,
        'is_rm_plus'   : False,
        'sampling_mode': 'c_sampling',
        'ave_mode'     : 'square',
        'log_mode'     : 'exponential',
        'op_env'       : 'PCFR'
    },
    # 'vanilla CFR': {
    #     'game'              : None,
    #     'rm_mode'           : 'vanilla',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'sampling_mode': 'no_sampling',
    #     'ave_mode'          : 'vanilla',
    #     'log_mode'          : 'exponential'
    # },
    'CFR+'                   : {
        'game'         : None,
        'rm_mode'      : 'vanilla',  # CFR
        'rm_eta'       : 1,
        'is_rm_plus'   : True,
        'sampling_mode': 'no_sampling',
        'ave_mode'     : 'liner',
        'log_mode'     : 'exponential'
    },
    'External-Sampling-MCCFR': {
        'game'         : None,
        'rm_mode'      : 'vanilla',  # CFR
        'rm_eta'       : 1,
        'is_rm_plus'   : False,
        'sampling_mode': 'c_sampling',
        'ave_mode'     : 'square',
        'log_mode'     : 'exponential'
    },
    # 'External-Sampling-MCCFR-log'        : {
    #     'game'              : None,
    #     'rm_mode'           : 'vanilla',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'sampling_mode': 'c_sampling',
    #     'ave_mode'          : 'log',
    #     'log_mode'          : 'exponential'
    # },
    # 'External-Sampling-MCCFR-liner': {
    #     'game'              : None,
    #     'rm_mode'           : 'vanilla',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'sampling_mode': 'c_sampling',
    #     'ave_mode'          : 'liner',
    #     'log_mode'          : 'exponential'
    # },
    # 'br-MCCFR-liner'               : {
    #     'game'              : None,
    #     'rm_mode'           : 'br',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'sampling_mode': 'c_sampling',
    #     'ave_mode'          : 'liner',
    #     'log_mode'          : 'exponential'
    # },
    # 'PCFR_log'                     : {
    #     'game'              : None,
    #     'rm_mode'           : 'br',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'sampling_mode': 'c_sampling',
    #     'ave_mode'          : 'log',
    #     'log_mode'          : 'exponential',
    #     'op_env'            : 'PCFR'
    # },
    # 'PCFR'                     : {
    #     'game'              : None,
    #     'rm_mode'           : 'br',  # CFR
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'sampling_mode': 'c_sampling',
    #     'ave_mode'          : 'vanilla',
    #     'log_mode'          : 'exponential',
    #     'op_env'            : 'PCFR'
    # },
    # 'External-Sampling-MCCFR-eta-fix': {
    #     'game'              : None,
    #     'rm_mode'           : 'eta_fix',  # CFR
    #     'rm_eta'            : 10,
    #     'is_rm_plus'        : False,
    #     'sampling_mode': 'c_sampling',
    #     'ave_mode'          : 'vanilla',
    #     'log_mode'          : 'exponential'
    # },

}

juqi_test_sampling_train_config = {
    # 'ES-MCCFR' : {
    #     'game'              : None,
    #     'rm_mode'           : 'vanilla',
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'sampling_mode': 'c_sampling',
    #     'ave_mode'          : 'vanilla',
    # },
    # 'PMCCFR' : {
    #     'game'              : None,
    #     'rm_mode'           : 'br',
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'sampling_mode': 'c_sampling',
    #     'ave_mode'          : 'vanilla',
    #     'op_env'            : 'PCFR'
    # },
    'CFR+': {
        'game'         : None,
        'rm_mode'      : 'vanilla',
        'rm_eta'       : 1,
        'is_rm_plus'   : True,
        'sampling_mode': 'no_sampling',
        'ave_mode'     : 'vanilla',
    },
    'PCFR': {
        'game'         : None,
        'rm_mode'      : 'br',
        'rm_eta'       : 1,
        'is_rm_plus'   : False,
        'sampling_mode': 'no_sampling',
        'ave_mode'     : 'vanilla',
        'op_env'       : 'PCFR'
    },
    # 'PCFR+'  : {
    #     'game'              : None,
    #     'rm_mode'           : 'br',
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : True,
    #     'sampling_mode': 'no_sampling',
    #     'ave_mode'          : 'vanilla',
    #     'op_env'            : 'PCFR'
    # },
    # 'PMCCFR+': {
    #     'game'              : None,
    #     'rm_mode'           : 'br',
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : True,
    #     'sampling_mode': 'c_sampling',
    #     'ave_mode'          : 'vanilla',
    #     'op_env'            : 'PCFR'
    # },
    'CFR' : {
        'game'         : None,
        'rm_mode'      : 'vanilla',
        'rm_eta'       : 1,
        'is_rm_plus'   : False,
        'sampling_mode': 'no_sampling',
        'ave_mode'     : 'square',
    },
    # 'ES-MCCFR' : {
    #     'game'              : None,
    #     'rm_mode'           : 'vanilla',
    #     'rm_eta'            : 1,
    #     'is_rm_plus'        : False,
    #     'sampling_mode': 'c_sampling',
    #     'ave_mode'          : 'square',
    # },
}
