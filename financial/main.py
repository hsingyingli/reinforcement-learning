import argparse
from tqdm import tqdm

from model import ModelFactory as mf
from framework import *
from env   import *


def main(args):
    config = mf.get_model(in_channels = args.feature_size, out_channels = args.feature_size, \
                                kernel_size = args.kernel_size, padding = args.padding, output_dimension = args.output_size)
    Agent = DDPG(args, config) 
    

    for step in tqdm(range(args.step)):
        # ------------------------ meta training --------------------
        total_loss = []
        for task in args.totoal_task:
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
    parser.add_argument('--device'            , type = str   , default = 'cuda')
    parser.add_argument('--step',  type = int, default = 100)


    args = parser.parse_args()
    main(args)