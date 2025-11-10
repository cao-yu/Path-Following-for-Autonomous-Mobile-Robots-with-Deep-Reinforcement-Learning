# -*- coding: utf-8 -*-

import math


DEFAULT_L = 0.2 # [m] lookahead distance
DEFAULT_K = 0.0 # proportional gain


def pure_pursuit_control(mdl, target_path, last_sn):
    # nearest point
    sn = target_path.find_nearest_point(last_sn, mdl.x, mdl.y)
    
    # look-ahead point  
    sl = sn + DEFAULT_L + DEFAULT_K * mdl.v
    if sl < target_path.len:
        tx, ty = target_path.X(sl), target_path.Y(sl)
    else:
        tx, ty = target_path.X(target_path.len), target_path.Y(target_path.len)
        sl = target_path.len
    
    # heading
    alpha = math.atan2(ty - mdl.y, tx - mdl.x) - mdl.yaw
    LA = mdl.calc_distance(tx, ty)
    omega = 2 * mdl.v * math.sin(alpha) / LA

    return omega, sn, sl
