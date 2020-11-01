import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--lr', type=float, default=7e-4)
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer alpha')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5)
    parser.add_argument(
        '--seed', type=int, default=1)
    parser.add_argument(
        '--num-steps',
        type=int,
        default=256,
        help='update freq')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='number of workers')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=20)
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100)
    parser.add_argument(
        '--env-name',
        default='Pong-v0')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/')
    parser.add_argument(
        '--log-dir',
        default='ac.log',
        help='directory to save log file')
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=False)

    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    return args
