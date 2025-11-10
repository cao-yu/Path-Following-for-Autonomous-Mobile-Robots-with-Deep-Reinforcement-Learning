# -*- coding: utf-8 -*-

import os


for i in range(5):
    os.system("python main.py \
              --policy {} \
              --seed {} \
	          --save_model"
              .format("SAC", i))
    
os.system("shutdown -s -t 60 ")