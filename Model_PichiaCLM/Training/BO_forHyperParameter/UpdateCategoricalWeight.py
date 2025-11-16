
#==========================================
#CoCaBO Algorithm - Function to faciitate sampling of categorical variable
#==========================================
import math
import collections
import pickle
import random

import GPy
import numpy as np
import pandas as pd
from tqdm import tqdm

from Kernel import MixtureViaSumAndProduct, CategoryOverlapKernel
from InitialData_Gen import initialize
from AcquisitionFunctions import EI, PI, UCB, AcquisitionOnSubspace


from scipy.optimize import minimize

from typing import Union, Tuple
from paramz.transformations import Logexp


def compute_reward_for_all_cat_variable(ht_next_batch_list,C_list,data, result, batch_size):
    # Obtain the reward for each categorical variable: B x len(self.C_list)
    ht_batch_list_rewards = np.zeros((batch_size, len(C_list)))
    for b in range(batch_size):
        ht_next_list = ht_next_batch_list[b, :]

        for i in range(len(ht_next_list)):
            idices = np.where(data[:, i] == ht_next_list[i])
            ht_result = result[idices]
            ht_reward = np.max(ht_result * -1)
            ht_batch_list_rewards[b, i] = ht_reward
    return ht_batch_list_rewards

def update_weights_for_all_cat_var(C_list, Gt_ht_list, ht_batch_list, Wc_list, gamma_list,
                                   probabilityDistribution_list, batch_size, S0=None):
    for j in range(len(C_list)):
        Wc = Wc_list[j]
        C = C_list[j]
        gamma = gamma_list[j]
        probabilityDistribution = probabilityDistribution_list[j]
        # print(f'cat_var={j}, prob={probabilityDistribution}')

        if batch_size > 1:
            print(j)
            ht_batch_list = ht_batch_list.astype(int)
            print(Gt_ht_list[:, j])
            Gt_ht = Gt_ht_list[:, j]
            mybatch_ht = ht_batch_list[:, j]  # 1xB
            for ii, ht in enumerate(mybatch_ht):
                Gt_ht_b = Gt_ht[ii]
                estimatedReward = 1.0 * Gt_ht_b / probabilityDistribution[ht]
                if ht not in S0:
                    Wc[ht] *= np.exp(batch_size * estimatedReward * gamma / C)
        else:
            Gt_ht = Gt_ht_list[j]
            ht = ht_batch_list[j]  # 1xB
            estimatedReward = 1.0 * Gt_ht / probabilityDistribution[ht]
            Wc[ht] *= np.exp(estimatedReward * gamma / C)

    return Wc_list