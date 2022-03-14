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

from env_mahjong import *


def test_performance(agents_test, x_test, o_test, a_test, m_test):

    acc = 0
    acc_random = 0  # Using random action
    steps_taken = 0

    agent_test = agents_test[0]

    # -------- Prediction Accuracy -----------
    x_e = x_test
    m_e = m_test
    a_e = a_test
    o_e = o_test

    for t in range(m_e.shape[0]):

        if m_e[t].sum() > 0:
            a_random = np.random.choice(m_e.shape[-1], p=np.float32(m_e[t]) / np.sum(np.float32(m_e[t])))
            if a_random == a_e[t]:
                acc_random += 1

            # action given by the agent
            a_rl = agent_test.select(x_e[t], o_e[t], action_mask=m_e[t], greedy=True)

            if a_rl == a_e[t]:
                acc += 1

            steps_taken += 1

    return acc_random / steps_taken, acc / steps_taken


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--max_all_steps', type=int, default=2000000, help="Number of total steps in the experiment")
parser.add_argument('--env', type=int, default=1, help="ID of Environment")
parser.add_argument('--type_id', type=int, default=1, help="ID of model type")
parser.add_argument('--seed', type=int, default=0, help="Random seed")

parser.add_argument('--beta', type=float, default=1e-5, help="Initial KLD coefficient")
parser.add_argument('--kld_target', type=float, default=50, help="KLD target")
parser.add_argument('--hidden_layer_width', type=int, default=1024, help="hidden layer width")
parser.add_argument('--verbose', type=float, default=1, help="Verbose")
parser.add_argument('--batch_size', type=int, default=1024, help="batch size")
parser.add_argument('--tau', type=float, default=1000, help="how many step to update target network once")

parser.add_argument('--opd_mu', type=float, default=0.01, help="mu in oracle policy distillation")
parser.add_argument('--cql', type=int, default=1, help="using CQL (not 0) or BC (0)")


args = parser.parse_args()

savepath = './data/'


if os.path.exists(savepath):
    logging.info('{} exists (possibly so do data).'.format(savepath))
else:
    os.makedirs(savepath)

# -------------------- Single trial function -------------------------


def run_vlog_single_trial(seed):
    # ----------------------------- Environment -----------------------------------------

    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.env == 1:
        env_name = "mahjong"
        env_test = EnvMahjong()
        gamma = 1.0
    else:
        raise ValueError

    # ----------------------- Hyperparameter and configurations -------------------------
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:{}".format(seed % torch.cuda.device_count()) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(seed % torch.cuda.device_count())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cql == 1:
        algorithm = "ddqn"
        alg_name = 'VLOG_CQL'
        alg_config = {}
        alg_config["cql_alpha"] = 1
        alg_config["dueling"] = True
    else:
        algorithm = "bc"
        alg_name = 'VLOG_BC'
        alg_config = {}

    type_list = ["vlog", "baseline", "oracle", "vlog-self", "suphx", "opd"]
    type = type_list[int(args.type_id) - 1]

    if type == "opd":
        opd_mu = args.opd_mu

        if algorithm == "ddqn":
            try:
                teacher_model = torch.load("./data/mahjong_VLOG_CQL_{}.model".format(int(seed)), map_location=device)
            except:
                print("Please train a oracle model before using OPD")
                quit(0)
        else:
            print("OPD does not make sense in BC")
            quit(0)
    else:
        opd_mu = 0
        teacher_model = None

    lr = 1e-4
    batch_size = int(args.batch_size)
    beta = args.beta
    kld_target = args.kld_target
    tau = args.tau
    hidden_layer_width = args.hidden_layer_width

    mahjong_augment = True

    max_all_steps = args.max_all_steps
    step_perf_eval = int(max_all_steps / 200)

    verbose = args.verbose
    #  ------------------ Initialize ---------------------

    agent = VLOG(env_test.observation_space, env_test.oracle_observation_space, env_test.action_space,
                 type=type, hidden_layer_width=hidden_layer_width,
                 algorithm=algorithm, alg_config=alg_config,
                 gamma=gamma, tau=tau, beta=beta, lr=lr, kld_target=kld_target,
                 verbose=verbose, device=device, opd_mu=opd_mu, opd_teacher_model=teacher_model)

    # --------------------- Load offline data ---------------
    data = sio.loadmat("./mahjong-all/mahjong-tenshou-offline-data-array-6dan-ver4-batch-0.mat")
    for key in ["V", "R", "A", "D"]:
        data[key] = data[key].reshape([-1])

    agents_test = [agent]

    total_len = data["V"].shape[0]
    n_batches = (total_len - 1) // batch_size
    used_len = int(n_batches * batch_size)
    permutation = np.random.permutation(used_len)

    XT, OT, AT, MT = data["X"][:1000], data["O"][:1000], data["A"][:1000], data["M"][:1000]
    # Test dataset to roughly estimate prediction acc. of the model.
    # Since we do not this the acc. as criteria, it is OK that the training dataset contains test dataset

    data["XP"] = torch.from_numpy(data["X"]).to(device=device)[permutation + 1]
    data["XP"] = data["XP"].view([n_batches, batch_size, *data["XP"].shape[1:]])

    data["OP"] = torch.from_numpy(data["O"]).to(device=device)[permutation + 1]
    data["OP"] = data["OP"].view([n_batches, batch_size, *data["OP"].shape[1:]])

    data["MP"] = torch.from_numpy(data["M"]).to(device=device)[permutation + 1]
    data["MP"] = data["MP"].view([n_batches, batch_size, *data["MP"].shape[1:]])

    for key in ["V", "X", "O", "A", "D", "R", "M"]:
        data[key] = torch.from_numpy(data[key]).to(device=device)[permutation]
        data[key] = data[key].view([n_batches, batch_size, *data[key].shape[1:]])

    batch_id = 0
    data_id = 0

    print("--------  preprocess finished  -------")

    KL = []

    clock_time_wrt_step = []

    acc_random_wrt_step = []
    acc_wrt_step = []

    global_steps = []

    global_step = 0

    start_time = time.time()

    while global_step < max_all_steps:

        if agent.type == "suphx":
            suphx_gamma = max(1 - 1.5 * global_step / max_all_steps, 0)
        else:
            suphx_gamma = None

        sample_and_training_start_time = time.time()

        if batch_id >= n_batches:  # load next dataset
            del data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                gc.collect()

            data_id = (data_id + 1) % 34

            data = sio.loadmat("/mahjong-all/mahjong-tenshou-offline-data-array-6dan-ver4-batch-{}.mat".format(data_id))
            for key in ["V", "R", "A", "D"]:
                data[key] = data[key].reshape([-1])
                gc.collect()

            total_len = data["V"].shape[0]
            n_batches = (total_len - 1) // batch_size
            used_len = int(n_batches * batch_size)
            permutation = np.random.permutation(used_len)

            data["XP"] = torch.from_numpy(data["X"]).to(device=device)[permutation + 1]
            data["XP"] = data["XP"].view([n_batches, batch_size, *data["XP"].shape[1:]])

            data["OP"] = torch.from_numpy(data["O"]).to(device=device)[permutation + 1]
            data["OP"] = data["OP"].view([n_batches, batch_size, *data["OP"].shape[1:]])

            data["MP"] = torch.from_numpy(data["M"]).to(device=device)[permutation + 1]
            data["MP"] = data["MP"].view([n_batches, batch_size, *data["MP"].shape[1:]])

            for key in ["V", "X", "O", "A", "D", "R", "M"]:
                data[key] = torch.from_numpy(data[key]).to(device=device)[permutation]
                data[key] = data[key].view([n_batches, batch_size, *data[key].shape[1:]])

            batch_id = 0

        if algorithm == 'ddqn':
            kld, loss_q, loss_a = agent.learn(X=data["X"][batch_id],
                                              XP=data["XP"][batch_id],
                                              O=data["O"][batch_id],
                                              OP=data["OP"][batch_id],
                                              A=data["A"][batch_id],
                                              R=data["R"][batch_id],
                                              D=data["D"][batch_id],
                                              V=data["V"][batch_id],
                                              action_masks=data["M"][batch_id],
                                              action_masks_tp1=data["MP"][batch_id],
                                              mahjong_augment=mahjong_augment,
                                              suphx_gamma=suphx_gamma)
        elif algorithm == 'bc':
            kld, loss_q, loss_a = agent.learn_bc(X=data["X"][batch_id],
                                                 XP=data["XP"][batch_id],
                                                 O=data["O"][batch_id],
                                                 OP=data["OP"][batch_id],
                                                 A=data["A"][batch_id],
                                                 R=data["R"][batch_id],
                                                 D=data["D"][batch_id],
                                                 V=data["V"][batch_id],
                                                 action_masks=data["M"][batch_id],
                                                 action_masks_tp1=data["MP"][batch_id],
                                                 mahjong_augment=mahjong_augment,
                                                 suphx_gamma=suphx_gamma)

        global_step += 1
        batch_id += 1

        if global_step < 10:
            print("sample and training time spend {} s".format(time.time() - sample_and_training_start_time))

        if global_step % step_perf_eval == 1:
            # ------------------ end reload data ----------------------
            acc_random, acc = test_performance(agents_test, XT, OT, AT, MT)

            try:
                KL.append(kld)
            except:
                KL.append(1)

            acc_random_wrt_step.append(acc_random)
            acc_wrt_step.append(acc)

            clock_time_wrt_step.append(time.time() - start_time)
            global_steps.append(global_step)
            logging.info(env_name + "seed {}".format(seed) + ": global step: {}, : prediction accuracy {})".format(
                global_step, acc))

    global_steps = np.reshape(global_steps, [-1]).astype(np.float64)
    clock_time_wrt_step = np.reshape(clock_time_wrt_step, [-1]).astype(np.float64)
    KL_wrt_step = np.reshape(KL, [-1]).astype(np.float64)

    data = {"minibatch_size": batch_size,
            "KL_wrt_step": KL_wrt_step,
            "acc_random_wrt_step": acc_random_wrt_step,
            "acc_wrt_step": acc_wrt_step,
            "clock_time_wrt_step": clock_time_wrt_step,
            "global_steps": global_steps}

    sio.savemat(savepath + env_name + "_" + alg_name + "_{}".format(seed) + ".mat", data, long_field_names=True)
    torch.save(agent, savepath + env_name + "_" + alg_name + "_{}".format(seed) + ".model")

# -------------------- Run experiment -------------------------


if __name__ == "__main__":
    seed = args.seed
    run_vlog_single_trial(seed)
