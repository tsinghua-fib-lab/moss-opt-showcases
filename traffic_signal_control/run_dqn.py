import argparse
import json
import os
import random
import sys
import time
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from moss import Engine, TlPolicy
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Env:
    def __init__(self, data_path, step_size, step_count):
        # create engine
        self.eng = Engine(
            map_file=f'{data_path}/map.bin',
            agent_file=f'{data_path}/agents.bin',
        )
        # the action size is equal to the phase count of each junction
        self.action_sizes = list(self.eng.get_junction_phase_counts())
        # save lane ids for observation
        self.in_lanes = self.eng.get_junction_inout_lanes()[0]
        # junction ids
        self.jids = range(len(self.action_sizes))
        # manual control
        self.eng.set_tl_policy_batch(self.jids, TlPolicy.MANUAL)
        print(f'Training on {len(self.action_sizes)} junctions')
        # observation size
        self.obs_sizes = [len(i) for i in self.in_lanes]
        # create engine checkpoint for reset
        self._cid = self.eng.make_checkpoint()
        self.step_size = step_size
        self.step_count = step_count
        self._step = 0
        self.info = {
            'ATT': 1e999,
            'Throughput': 0,
            'reward': 0
        }

    def reset(self):
        # reset engine
        self.eng.restore_checkpoint(self._cid)

    def observe(self):
        # observation is normalized vehicle count
        cnt = np.minimum(1, self.eng.get_lane_vehicle_counts()/50)
        return [cnt[i] for i in self.in_lanes]

    def step(self, action):
        # apply action
        for i, j in zip(self.jids, action):
            self.eng.set_tl_phase(i, j)
        # next step
        self.eng.next_step(self.step_size)
        # state
        s = self.observe()
        cnt = np.minimum(1, self.eng.get_lane_vehicle_counts()/50)
        # reward
        r = [-np.mean(cnt[i]) for i in self.in_lanes]
        self.info['reward'] = np.mean(r)
        self._step += 1
        done = False
        # reset
        if self._step >= self.step_count:
            self.info['ATT'] = self.eng.get_finished_vehicle_average_traveling_time()
            self.info['Throughput'] = self.eng.get_finished_vehicle_count()
            self._step = 0
            self.reset()
            done = True
        return s, r, done, self.info


class Replay:
    def __init__(self, max_size):
        self._data = deque([], max_size)

    def add(self, s, a, r, sp, d):
        self._data.append([s, a, r, sp, d])

    def sample(self, batchsize, transpose=False):
        s, a, r, sp, d = zip(*random.sample(self._data, min(len(self._data), batchsize)))
        if transpose:
            s, a, r, sp = (list(zip(*i)) for i in [s, a, r, sp])
        return s, a, r, sp, d


def make_mlp(*sizes, act=nn.GELU, dropout=0.1):
    layers = []
    for i, o in zip(sizes, sizes[1:]):
        layers.append(nn.Linear(i, o))
        layers.append(act())
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers[:-2])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def lerp(a, b, t):
    t = min(1, t)
    return a*(1-t)+b*t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/hangzhou')
    parser.add_argument('--steps', type=int, default=3600)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--training_step', type=int, default=50000)
    parser.add_argument('--training_start', type=int, default=2000)
    parser.add_argument('--training_freq', type=int, default=10)
    parser.add_argument('--target_freq', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--buffer_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mlp', type=str, default='256,256')
    args = parser.parse_args()

    print('Moss version:', Engine.__version__)

    # create log path
    path = time.strftime('log/%Y%m%d-%H%M%S')
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/cmd.sh', 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\ntensorboard --port 8888 --logdir '+os.path.abspath(path))
    with open(f'{path}/args.json', 'w') as f:
        json.dump(vars(args), f)
    print('tensorboard --port 8888 --logdir '+os.path.abspath(path))

    writer = SummaryWriter(path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    device = torch.device("cuda")

    # create environment
    env = Env(
        data_path=args.data,
        step_size=args.interval,
        step_count=args.steps//args.interval,
    )
    # create agents
    Q = [make_mlp(o, *(int(i) for i in args.mlp.split(',')), a).to(device) for o, a in zip(env.obs_sizes, env.action_sizes)]
    opt = optim.AdamW([j for i in Q for j in i.parameters()], lr=args.lr)
    Q_target = deepcopy(Q)
    obs = env.observe()
    replay = Replay(args.buffer_size)
    with tqdm(range(args.training_step), ncols=100, smoothing=0.1) as bar:
        for step in bar:
            _st = time.time()
            eps = lerp(1, 0.05, step/10000)
            action = []
            for q, o, a in zip(Q, obs, env.action_sizes):
                if step < args.training_start or random.random() < eps:
                    # explore
                    action.append(random.randint(0, a-1))
                else:
                    # exploit
                    action.append(
                        torch.argmax(q(torch.tensor(o, dtype=torch.float32, device=device))).item()
                    )
            next_obs, reward, done, info = env.step(action)
            if done:
                writer.add_scalar('metric/ATT', info['ATT'], step)
                writer.add_scalar('metric/Throughput', info['Throughput'], step)
            writer.add_scalar('metric/Reward', info['reward'], step)
            replay.add(obs, action, reward, next_obs, done)
            obs = next_obs
            # training
            if step >= args.training_start and step % args.training_freq == 0:
                s, a, r, sp, d = replay.sample(args.batchsize, transpose=True)
                d = torch.tensor(d, dtype=torch.float32, device=device)
                loss = 0
                for q, qt, s, a, r, sp in zip(Q, Q_target, s, a, r, sp):
                    s = torch.tensor(np.array(s), dtype=torch.float32, device=device)
                    a = torch.tensor(a, dtype=torch.long, device=device)
                    r = torch.tensor(r, dtype=torch.float32, device=device)
                    sp = torch.tensor(np.array(sp), dtype=torch.float32, device=device)
                    with torch.no_grad():
                        y_target = r+args.gamma*qt(sp).max(1).values*(1-d)
                    y = q(s).gather(-1, a[..., None]).view(-1)
                    loss = loss+F.mse_loss(y, y_target)
                loss = loss/len(Q)
                opt.zero_grad()
                loss.backward()
                opt.step()
                writer.add_scalar('chart/loss', loss.item(), step)
                bar.set_description(f'ATT: {info["ATT"]:.3f} TP: {info["Throughput"]} loss: {loss.item():.6f}')
                if step % args.target_freq == 0:
                    for a, b in zip(Q, Q_target):
                        b.load_state_dict(a.state_dict())
            writer.add_scalar('chart/FPS', 1/(time.time()-_st), step)
    writer.close()


if __name__ == '__main__':
    main()
