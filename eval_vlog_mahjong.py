import gc
import os
import time
import sys
import logging
import warnings
from models import VLOG
import numpy as np
import scipy.io as sio
import torch
import argparse

import MahjongPy as mp
from env_mahjong import *


def test_performance(agents_test, env_test, times=80, use_posterior=None, rotate_player=False):
    if use_posterior is None:
        use_posterior = [False, False, False, False]

    if times % 4 != 0:
        warnings.warn("For fairness, times should be 4N where N is positive integer so that each player acts as Oya for the same times!")

    assert len(agents_test) == 4
    steps_taken = 0
    max_steps = 1000
    # -------------- Return vs. pretrained player -------------

    game = 0
    trials = 0
    payoffs_array = []
    winning_counts = []
    deal_in_counts = []
    while game < times:
        trials += 1

        if rotate_player:
            env_test.reset(0, 'east')
        else:
            env_test.reset(game % 4, 'east')

        payoffs = np.zeros([4])

        for tt in range(max_steps):

            curr_pid = env_test.get_curr_player_id()
            valid_actions = env_test.get_valid_actions(nhot=False)

            if rotate_player:
                agent_id = (curr_pid + game) % 4
            else:
                agent_id = curr_pid

            action_mask = env_test.get_valid_actions(nhot=True)

            a = agents_test[agent_id].select(env_test.get_obs(curr_pid), env_test.get_oracle_obs(curr_pid),
                                             action_mask, greedy=True,
                                             use_posterior=use_posterior[agent_id])

            sp, r, done, _ = env_test.step(curr_pid, a)

            steps_taken += 1

            if env_test.has_done():

                if rotate_player:
                    id_shift = np.array([(0 - game) % 4, (1 - game) % 4, (2 - game) % 4, (3 - game) % 4], dtype=int)
                    payoffs = payoffs + np.array(env_test.get_payoffs())[id_shift]
                else:
                    payoffs = payoffs + np.array(env_test.get_payoffs())

                print("payoffs of game {} :".format(game), payoffs)

                curr_wins = np.zeros([4])
                curr_deal_ins = np.zeros([4])

                if env_test.t.get_result().result_type == mp.ResultType.RonAgari:
                    for ii in range(4):  # consider multiple players Agari
                        if payoffs[ii] > 0:
                            curr_wins[ii] = 1
                    curr_deal_ins[np.argmin(payoffs)] = 1
                elif env_test.t.get_result().result_type == mp.ResultType.TsumoAgari:
                    curr_wins[np.argmax(payoffs)] = 1

                payoffs_array.append(payoffs)
                winning_counts.append(curr_wins)
                deal_in_counts.append(curr_deal_ins)
                game += 1
                break

    print("Test {} games".format(times))

    return np.array(payoffs_array), steps_taken, times, np.array(winning_counts), np.array(deal_in_counts)


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=int, default=1, help="ID of Environment")

parser.add_argument('--model_dir_0', type=str, help="Directory of the model for player 0 on the mahjong table")
parser.add_argument('--model_dir_1', type=str, help="Directory of the model for player 1 on the mahjong table")
parser.add_argument('--model_dir_2', type=str, help="Directory of the model for player 2 on the mahjong table")
parser.add_argument('--model_dir_3', type=str, help="Directory of the model for player 3 on the mahjong table")

parser.add_argument('--seed', type=int, default=0, help="Random seed")

parser.add_argument('--times', type=int, default=100, help="number of games tested")

args = parser.parse_args()

savepath = './data/'

if os.path.exists(savepath):
    logging.info('{} exists (possibly so do data).'.format(savepath))
else:
    os.makedirs(savepath)

seed = args.seed

np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.device_count() > 1:
    device = torch.device("cuda:{}".format(seed % torch.cuda.device_count()) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(seed % torch.cuda.device_count())
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rotate_player = True

model_dirs = [args.model_dir_0,
              args.model_dir_1,
              args.model_dir_2,
              args.model_dir_3]

test_model_dirs = [model_dirs[0],
                   model_dirs[1],
                   model_dirs[2],
                   model_dirs[3]]

agents_test = [torch.load(model_dir, map_location=torch.device(device)) for model_dir in test_model_dirs]

for ag in agents_test:
    ag.device = device

env_name = "mahjong"
env_test = EnvMahjong()

results_array, total_steps, n_games, winnings, dealins = test_performance(
    agents_test, env_test, times=args.times, rotate_player=rotate_player)

data = {"results": results_array,
        "winnings": winnings,
        "dealins": dealins,
        "total_steps": total_steps,
        "n_games": n_games,
        "model_dirs": model_dirs}

sio.savemat(savepath + env_name + "_" + "evaluation" + "_{}".format(seed) + ".mat", data, long_field_names=True)
