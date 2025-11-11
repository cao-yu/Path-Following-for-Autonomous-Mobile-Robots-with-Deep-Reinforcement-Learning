# -*- coding: utf-8 -*-
#
# ATTENTION ！
# In dissertation, straight lines are omitted in performance test.
# For running the script
# go to "param_fcn.py"
# comment the following code
#
# if self.ep_cnt % 10 == 0:
#
# to pass the generation of straight line
#

import sys
from pathlib import Path
base = Path(__file__).resolve()
for i in range(3):
    p = base.parents[i]
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from kinematics import Unicycle
from reference_path import ArcReferencePath
from control_law_arc import pure_pursuit_control


T = 400.0  # max simulation steps
num = 1000 # Number of attempts at each speed
v_array = np.arange(0.1, 0.41, 0.05) # [m/s]

# cross-track error
cte_tol = np.array([0.1, 0.2, 0.3])



if __name__ == "__main__":
    # run following script
    if 0: 
        
        # reference path 
        rpath = ArcReferencePath("random")
        rpath.seed(100)
        
        # count failure number
        failure_num = np.zeros((len(v_array), len(cte_tol)))
        
        # count completation rate
        completion_rate = np.zeros((num, len(v_array), len(cte_tol)))
            
        for i in tqdm(range(num)):
            # new path
            rpath.gen_rand_path()
                
            for j in range(len(v_array)): 
                # vehicle model
                mdl = Unicycle()
                mdl.seed(100)
                mdl.reset(yaw=rpath.calc_yaw(0.0), flag="random") # re-initialize randomly
          
                # initialize lookahead point
                omega_ref, sn, sl = pure_pursuit_control(mdl, rpath, 0.01) 
                
                cnt = 0
                k_cte = 0
                
                while True:
                        
                    cnt += 1
                    
                    # update      
                    mdl.update(v_array[j], omega_ref)
                         
                    # next time step
                    omega_ref, sn, sl = pure_pursuit_control(mdl, rpath, sn)   
                    
                    # compute errors
                    cte, _ = rpath.calc_error(mdl.x, mdl.y, mdl.yaw, sn)
                    
                    # terminated
                    if abs(cte) >= cte_tol[k_cte]: 
                        failure_num[j][k_cte] += 1
                        completion_rate[i][j][k_cte] = sn / rpath.len
                        k_cte += 1
                        if k_cte == len(cte_tol):
                            break
                    
                    # reach time limit or goal
                    if T <= cnt or rpath.len - sn < 0.02:
                        while k_cte < len(cte_tol):
                            completion_rate[i][j][k_cte] = sn / rpath.len
                            k_cte += 1
                        break
                    
        failure = failure_num / num       
        np.save("./failure_rate", failure) 
        np.save("./completion_rate",completion_rate)
    
    # load from saved files
    else: 
        failure = np.load("./failure_rate.npy") 
        completion_rate = np.load("./completion_rate.npy")
        
    avg_completion = np.mean(completion_rate, axis=0) 
    std_completion = np.std(completion_rate, axis=0) 

    # plotting
    plt.style.use('classic')
    plt.rcParams["font.size"] = 20

    for k in range(len(cte_tol)):
        fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
        fig.patch.set_alpha(0.0)
   
        ax.plot(
            v_array, failure[:,k], 
            'b-o', linewidth=2.5, markeredgewidth=3, ms=15, 
            label="Failure")
        ax.plot(
            v_array, avg_completion[:,k], 
            'r-o', linewidth=2.5, markeredgewidth=3, ms=15, 
            label="Completion")
        ax.fill_between(
            v_array, 
            avg_completion[:,k] - std_completion[:,k], 
            avg_completion[:,k] + std_completion[:,k], 
            fc="r", ec="white", alpha=0.5) # confidence bands
        
        ax.set_xlabel("Velocity [m/s]", fontsize=20, labelpad=10)
        ax.set_ylabel("Average Rate", fontsize=20)
        ax.set_title(r'$\left| e_p \right| < {:.1f}$ m'.format(cte_tol[k]), fontsize=26, y=1.025)
    
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlim(0.09, 0.41)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks(v_array)
        ax.set_xlabel
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.legend(loc='upper left', numpoints=1, fontsize=18)

        plt.savefig(f'cte_{k}.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.show()
        plt.close(fig)