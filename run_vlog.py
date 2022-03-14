import os
import time
import logging
import warnings
from models import VLOG
import numpy as np
import scipy.io as sio
import torch
import argparse


def test_performance(agent_test, env_test, max_steps, times=5):
    EpiTestRet = 0
    steps_taken = 0

    for _ in range(times):

        # reset each episode
        sp = env_test.reset()
        agent_test.init_states()

        for _ in range(max_steps):

            a = agent_test.select(sp, env_test.get_oracle_obs())
            if np.any(np.isnan(a)):
                raise ValueError

            sp, r, done, _ = env_test.step(a)

            EpiTestRet += r
            steps_taken += 1

            if done:
                break

    EpiTestRet_greedy = 0
    steps_taken_greedy = 0

    for _ in range(times):

        # reset each episode
        sp = env_test.reset()
        agent_test.init_states()

        for _ in range(max_steps):

            a = agent_test.select(sp, env_test.get_oracle_obs(), greedy=True)
            if np.any(np.isnan(a)):
                raise ValueError

            sp, r, done, _ = env_test.step(a)

            EpiTestRet_greedy += r
            steps_taken_greedy += 1

            if done:
                break

    return EpiTestRet / times, EpiTestRet_greedy / times, steps_taken / times, steps_taken_greedy / times


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('--env', type=int, help="ID of Environment")
parser.add_argument('--max_all_steps', type=int, default=0, help="Number of total steps in the experiment")
parser.add_argument('--type_id', type=int, default=1, help="ID of model type")
parser.add_argument('--seed', type=int, default=0, help="Random seed")

parser.add_argument('--beta', type=float, default=1e-5, help="Initial KLD coefficient")
parser.add_argument('--kld_target', type=float, default=50, help="KLD target")
parser.add_argument('--epsilon', type=float, default=0.1, help="epsilon-greedy")
parser.add_argument('--tau', type=float, default=1000, help="target network update interval")
parser.add_argument('--hidden_layer_width', type=int, default=256, help="hidden layer width")
parser.add_argument('--verbose', type=float, default=1, help="Verbose")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument('--rl_interval', type=int, default=4, help="how many environment step every RL gradient step")

parser.add_argument('--opd_mu', type=float, default=10, help="mu in oracle policy distillation")

args = parser.parse_args()

savepath = './data/'

if os.path.exists(savepath):
    logging.info('{} exists (possibly so do data).'.format(savepath))
else:
    os.makedirs(savepath)


def run_vlog_single_trial(seed):
    # ----------------------------- Environment -----------------------------------------

    # use standard Gym env, except that it should additionally have an attribute "oracle_observation space" and a method
    # "get_oracle_obs()" for oracle information.

    np.random.seed(seed)
    torch.manual_seed(seed)

    broken_pixel_mask = None
    broken_pixels_ratio = 0.125

    if args.env == 1:
        env_name = "breakout"
        if args.max_all_steps == 0:
            max_all_steps = 5000000
        else:
            max_all_steps = args.max_all_steps

    elif args.env == 2:
        env_name = "seaquest"
        if args.max_all_steps == 0:
            max_all_steps = 8000000
        else:
            max_all_steps = args.max_all_steps

    elif args.env == 3:
        env_name = "space_invaders"
        if args.max_all_steps == 0:
            max_all_steps = 5000000
        else:
            max_all_steps = args.max_all_steps

    elif args.env == 4:
        env_name = "freeway"
        if args.max_all_steps == 0:
            max_all_steps = 5000000
        else:
            max_all_steps = args.max_all_steps

    elif args.env == 5:
        env_name = "asterix"
        if args.max_all_steps == 0:
            max_all_steps = 5000000
        else:
            max_all_steps = args.max_all_steps

    if 0 < args.env <= 5:
        from minatar_env import MinAtarEnv
        env = MinAtarEnv(env_name, broken_pixel_mask=broken_pixel_mask, broken_pixels_ratio=broken_pixels_ratio)
        env_test = MinAtarEnv(env_name, broken_pixel_mask=broken_pixel_mask, broken_pixels_ratio=broken_pixels_ratio)
        gamma = 0.995
        max_steps = 5000
        obs_uint8 = True

    if args.env == 0:
        from tasks import SimpleMaze
        env = SimpleMaze()
        env_test = SimpleMaze()
        env_name = "maze"
        gamma = 0.995
        max_steps = 5000
        obs_uint8 = False
        if args.max_all_steps == 0:
            max_all_steps = 4000000
            args.rl_interval = 4
        else:
            max_all_steps = args.max_all_steps

    # ----------------------- Hyperparameter and configurations -------------------------

    type_list = ["vlog", "baseline", "oracle", "vlog-self", "suphx", "opd"]
    assert 1 <= args.type_id <= 6
    type = type_list[int(args.type_id) - 1]

    print("******************  Current Model Type is ", type)

    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:{}".format(seed % torch.cuda.device_count()) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(seed % torch.cuda.device_count())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if type == "opd":
        opd_mu = float(args.opd_mu)
        try:
            teacher_model = torch.load("./data/{}_VLOG_DDQN_{}.model".format(env_name, seed), map_location=device)
        except:
            print("Please train a oracle model before using OPD")
            quit(0)
    else:
        opd_mu = 0
        teacher_model = None

    algorithm = "ddqn"
    alg_name = 'VLOG_DDQN'
    alg_config = {}
    alg_config["dueling"] = True
    tau = args.tau

    # Episilon greedy for DDQN only
    epsilon = args.epsilon
    epsilon_decay = 1
    epsilon_min = args.epsilon

    batch_size = args.batch_size
    lr = 1e-4

    # max_buffer_episode_size = np.inf

    hidden_layer_width = args.hidden_layer_width

    beta = args.beta
    kld_target = args.kld_target

    step_start_rl = 50000
    max_buffer_size = max_all_steps

    random_before_rl_start = True

    train_step_rl = args.rl_interval  # how many times of RL training after step_start_rl
    # max_all_steps = args.max_all_steps

    step_perf_eval = max(int(max_all_steps / 200), max_steps)  # how many steps to do evaluation
    verbose = args.verbose

    #  ------------------ Initialize ---------------------

    agent = VLOG(env.observation_space, env.oracle_observation_space, env.action_space,
                 type=type, hidden_layer_width=hidden_layer_width,
                 algorithm=algorithm, alg_config=alg_config, epsilon=epsilon,
                 gamma=gamma, tau=tau, beta=beta, lr=lr, kld_target=kld_target,
                 verbose=verbose, device=device, opd_teacher_model=teacher_model, opd_mu=opd_mu)

    # Replay Buffers
    if obs_uint8:
        X = np.zeros([max_buffer_size, *env.observation_space.shape], dtype=np.uint8)
        O = np.zeros([max_buffer_size, *env.oracle_observation_space.shape], dtype=np.uint8)
    else:
        X = np.zeros([max_buffer_size, *env.observation_space.shape], dtype=np.float32)
        O = np.zeros([max_buffer_size, *env.oracle_observation_space.shape], dtype=np.float32)

    A = np.zeros([max_buffer_size], dtype=np.int64)
    R = np.zeros([max_buffer_size], dtype=np.float32)
    D = np.zeros([max_buffer_size], dtype=np.float32)
    V = np.zeros([max_buffer_size], dtype=np.float32)

    info_final = []

    L = []
    episode_reward = []
    KL = []

    performance_wrt_step = []
    steps_taken_wrt_step = []
    performance_greedy_action_wrt_step = []
    steps_taken_greedy_action_wrt_step = []
    clock_time_wrt_step = []
    global_steps = []

    zp_final = []

    e_real = 0
    global_step = 0

    start_time = time.time()

    while global_step < max_all_steps - 1:

        sp = env.reset()
        agent.init_states()  # Must be done for RNN VLOG!

        X[global_step % max_buffer_size] = sp
        O[global_step % max_buffer_size] = env.get_oracle_obs()

        for t in range(max_steps):

            if random_before_rl_start and global_step < step_start_rl:
                a = np.random.randint(0, env.action_space.n)
            else:
                a = agent.select(sp, env.get_oracle_obs())

            sp, r, done, info = env.step(a)

            if global_step >= max_all_steps - 2000:
                zp_final.append(agent.zp_tm1.detach().cpu().numpy())
                if isinstance(info, np.ndarray):
                    info_final.append(info)

            X[(global_step + 1) % max_buffer_size] = sp
            O[(global_step + 1) % max_buffer_size] = env.get_oracle_obs()
            A[global_step % max_buffer_size] = a
            R[global_step % max_buffer_size] = r
            D[global_step % max_buffer_size] = done
            V[global_step % max_buffer_size] = 1

            global_step += 1

            if global_step >= max_all_steps - 2:
                break

            if global_step > step_start_rl:

                # env.render()
                if agent.type == "suphx":
                    suphx_gamma = max(1 - 1.5 * global_step / max_all_steps, 0)
                else:
                    suphx_gamma = None

                if algorithm == 'ddqn':
                    agent.epsilon = (agent.epsilon - epsilon_min) * epsilon_decay + epsilon_min

                if global_step % train_step_rl == 0:
                    sampled_steps = np.random.choice(min(global_step - 2, max_buffer_size - 2), batch_size)
                    loss_z, loss_q, loss_a = agent.learn(X=X[sampled_steps],
                                                         XP=X[sampled_steps + 1],
                                                         O=O[sampled_steps],
                                                         OP=O[sampled_steps + 1],
                                                         A=A[sampled_steps],
                                                         R=R[sampled_steps],
                                                         D=D[sampled_steps],
                                                         V=V[sampled_steps],
                                                         suphx_gamma=suphx_gamma)
            if global_step % step_perf_eval == 0:
                try:
                    KL.append(loss_z)
                except:
                    KL.append(1)
                EpiTestRet, EpiTestRet_greedy, steps_taken, steps_taken_greedy_action = test_performance(
                    agent, env_test, max_steps=max_steps, times=10)
                performance_wrt_step.append(EpiTestRet)
                steps_taken_wrt_step.append(steps_taken)
                performance_greedy_action_wrt_step.append(EpiTestRet_greedy)
                steps_taken_greedy_action_wrt_step.append(steps_taken_greedy_action)
                clock_time_wrt_step.append(time.time() - start_time)
                global_steps.append(global_step)
                logging.info(env_name + "seed {}".format(
                    seed) + ": global step: {}, : steps {}, test return {} , {} (greedy)".format(
                    global_step, t, EpiTestRet, EpiTestRet_greedy))

            if done or t == max_steps - 1:

                episode_length = int(t + 1)
                L.append(episode_length)
                episode_reward.append(np.sum(R[global_step - episode_length:global_step]))

                break

        if args.verbose:
            print(env_name + "seed {}".format(seed) + " -- episode {} : steps {}, total reward {}".format(
                  e_real, t, episode_reward[-1]))

        e_real += 1

    performance_wrt_step = np.reshape(performance_wrt_step, [-1]).astype(np.float64)
    steps_taken_wrt_step = np.reshape(steps_taken_wrt_step, [-1]).astype(np.float64)
    performance_greedy_action_wrt_step = np.reshape(performance_greedy_action_wrt_step, [-1]).astype(np.float64)
    steps_taken_greedy_action_wrt_step = np.reshape(steps_taken_greedy_action_wrt_step, [-1]).astype(np.float64)
    global_steps = np.reshape(global_steps, [-1]).astype(np.float64)
    clock_time_wrt_step = np.reshape(clock_time_wrt_step, [-1]).astype(np.float64)
    KL_wrt_step = np.reshape(KL, [-1]).astype(np.float64)
    episode_reward = np.reshape(episode_reward, [-1]).astype(np.float64)

    data = {"max_steps": max_steps,
            "step_start_rl": step_start_rl,
            "minibatch_size": batch_size,
            "train_step_rl": train_step_rl,
            "steps": L,
            "episode_reward": episode_reward,
            "KL_wrt_step": KL_wrt_step,
            "performance_wrt_step": performance_wrt_step,
            "steps_taken_wrt_step": steps_taken_wrt_step,
            "performance_greedy_action_wrt_step": performance_greedy_action_wrt_step,
            "steps_taken_greedy_action_wrt_step": steps_taken_greedy_action_wrt_step,
            "clock_time_wrt_step": clock_time_wrt_step,
            "global_steps": global_steps}

    if len(info_final):
        try:
            data["zp_final"] = np.array(zp_final)
            data["info_final"] = np.array(info_final)
        except:
            pass

    sio.savemat(savepath + env_name + "_" + alg_name + "_{}".format(seed) + ".mat", data, long_field_names=True)

    # save the model
    torch.save(agent, savepath + env_name + "_" + alg_name + "_{}".format(seed) + ".model")

# -------------------- Run experiment -------------------------


if __name__ == "__main__":
    seed = args.seed
    run_vlog_single_trial(seed)
