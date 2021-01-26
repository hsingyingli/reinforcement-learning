import argparse 
from framework import *


def main(args):
    agent = DDPG(args)
    # agent.env = gym.wrappers.Monitor(agent.env, directory="mon", force=True)
    print('Game Start')
    agent.solve()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env'               , type = str   , default = 'MountainCarContinuous-v0')
    parser.add_argument('--device'            , type = str   , default = 'cuda')
    parser.add_argument('--critic_lr'         , type = float , default = 0.001)
    parser.add_argument('--actor_lr'          , type = float , default = 0.0001)
    parser.add_argument('--tau'               , type = float , default = 0.001)
    parser.add_argument('--gamma'             , type = float , default = 0.99)
    parser.add_argument('--epsilon'           , type = float , default = 1.)
    parser.add_argument('--deepsilon'         , type = float , default = 0.95)
    parser.add_argument('--batch_size'        , type = int   , default = 16)
    parser.add_argument('--max_episode'       , type = int   , default = 1)
    parser.add_argument('--max_step'          , type = int   , default = 2000000)
    parser.add_argument('--hidden_size'       , type = int   , default = 128)
    parser.add_argument('--capacity'          , type = int   , default = 30)
    
    
    args = parser.parse_args()
    print(args)
    main(args)