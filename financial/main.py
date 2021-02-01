import argparse
import os
from tqdm import tqdm

from model import ModelFactory as mf
from framework import *
from env   import *


def main(args):
    config = mf.get_model(in_channels = args.feature_size, out_channels = args.feature_size, \
                                kernel_size = args.kernel_size, padding = args.padding, output_dimension = args.output_size)
    print('---------- Load model config ----------')
    Agent = DDPG(args, config) 
    print('---------- Initialize  Agent ----------')
    # total_task = os.listdir('.data')
    total_task = ['CSI300.csv', 'S&P500.csv']
    for step in tqdm(range(args.step)):
        # ------------------------ meta training --------------------
        total_loss = []
        for task in total_task:
            # -----------------   switch to new environment (new task) ---------------------
            Agent.env     = environment(stock = task, date = args.date, time_step = args.time_step)
            # initial exploration rate
            Agent.epsilon = args.epsilon
            # learn to solve 
            loss = Agent.solve()
            total_loss.append(loss)

        # --------------update representation layer (meta knowledge)--------------------
        Agent.update_representation(total_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device'            , type = str      , default = 'cpu')
    parser.add_argument('--date'              , type = str      , default = '2010-01-01')
    parser.add_argument('--step'              , type = int      , default = 100)
    parser.add_argument('--feature_size'      , type = int      , default = 16)
    parser.add_argument('--hidden_size'      , type = int      , default = 128)
    parser.add_argument('--output_size'       , type = int      , default = 3)
    parser.add_argument('--time_step'         , type = int      , default = 22)
    parser.add_argument('--batch_size'        , type = int      , default = 100)
    parser.add_argument('--max_episode'       , type = int      , default = 200)
    parser.add_argument('--capacity'          , type = int      , default = 10000)
    parser.add_argument('--critic_lr'         , type = float    , default = 0.001)
    parser.add_argument('--actor_lr'          , type = float    , default = 0.001)
    parser.add_argument('--meta_lr'           , type = float    , default = 0.001)
    parser.add_argument('--tau'               , type = float    , default = 0.001)
    parser.add_argument('--gamma'             , type = float    , default = 0.9)
    parser.add_argument('--epsilon'           , type = float    , default = 1.)
    parser.add_argument('--deepsilon'         , type = float    , default = 0.995)
    parser.add_argument('--kernel_size'       , nargs = '+' ,type = int      , default = [8, 5, 3])
    parser.add_argument('--padding'           , nargs = '+' ,type = int      , default = [4, 2, 1])
    args = parser.parse_args()
    main(args)
