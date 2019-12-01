from torch.utils.tensorboard import SummaryWriter
import argparse
import subprocess
import multiprocessing
import agent
import env

parser = argparse.ArgumentParser()
parser.add_argument('--env')
#learning algorithm to be used
parser.add_argument('--alg')
parser.add_argument('--tensorboard')
parser.add_argument('--mode', '-m')
parser.add_argument('--model')
#no training episodes
parser.add_argument('--eps')

parser.add_argument('--render')

args = parser.parse_args()


if args.tensorboard:
    writer = SummaryWriter()

    write_proc = subprocess.Popen(['tensorboard', '--logdir', '{}'.format(args.tensorboard)])

env = env.Environment(args.env)

if args.alg == 'DQN':
    agent = agent.DQNAgent(env, args.mode, args.model, writer)

try:
    if args.mode == 'train':
        agent.train(int(args.eps), args.render)
    elif args.mode == 'play':
        agent.play(int(args.eps))
except KeyboardInterrupt:
    print('PROCESS KILLED BY USER')
finally:
    env.close()
    if args.tensorboard:
        write_proc.terminate()
