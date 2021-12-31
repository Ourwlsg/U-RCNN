#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@IDE    ：PyCharm
@Author ：zhucc
@Date   ：2021-12-7 16:06
@Desc   ：
==================================================
"""
import os
import sys
import time
import numpy as np


def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    memory = [int(gpu_status[2].split('/')[0].split('M')[0].strip()),
              int(gpu_status[6].split('/')[0].split('M')[0].strip()),
              int(gpu_status[10].split('/')[0].split('M')[0].strip()),
              int(gpu_status[14].split('/')[0].split('M')[0].strip())]

    # print(memory[0])
    # print(memory[1])
    # print(memory[2])
    # print(memory[3])
    # gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return memory


def narrow_setup(interval=5):
    while True:
        gpu_memory = gpu_info()

        # sum_gpu_memory = sum(gpu_memory)

        # for i, m in enumerate(gpu_memory):  # set waiting condition
        #     # symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        #     # gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        #     # sys.stdout.write('\r' + gpu_memory_str + ' ' + symbol)
        #
        #     if m < 500:
        #         return i
        m = np.array(gpu_memory)
        g = np.argwhere(m < 500)
        if len(g) >= 2:
            g = g[:2].tolist()
            gpu_id = repr(g).replace('[', '', -1).replace(']', '', -1)
            return gpu_id
        time.sleep(interval)


if __name__ == '__main__':
    gpu_id = narrow_setup()
    print(f"GPU ID is {gpu_id} ,start running......")
    cmd = f'python train.py'
    os.system(cmd)
