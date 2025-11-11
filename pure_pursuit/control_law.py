# -*- coding: utf-8 -*-

import math

"""
Ref:
    https://arxiv.org/pdf/1604.07446.pdf
    
"""


DEFAULT_L = 0.1 # [m] lookahead distance
DEFAULT_K = 0.4 # proportional gain

def pure_pursuit_control(mdl, target_path, last_sl, L=DEFAULT_L):
    # look-ahead point
    LA = L + DEFAULT_K * mdl.v
    sl = target_path.find_lookahead_point(last_sl, mdl.x, mdl.y, LA)

    if last_sl > sl:
        sl = last_sl
        
    if sl < target_path.len:
        tx, ty = target_path.X(sl), target_path.Y(sl)
    else:
        tx, ty = target_path.X(target_path.len), target_path.Y(target_path.len)
        sl = target_path.len

    alpha = math.atan2(ty - mdl.y, tx - mdl.x) - mdl.yaw
    omega = 2 * mdl.v * math.sin(alpha) / LA

    return omega, sl
