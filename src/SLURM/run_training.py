import os
import logging, argparse
import itertools

import platform

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")

    parser.add_argument("--odir", default = "None")
    parser.add_argument("--tag", default = "None")

    parser.add_argument("--idir", default = "None")

    parser.add_argument("--indices", default="None")

    parser.add_argument("--data", default = "None")

    parser.add_argument("--architecture", default="architectures/var_size_two_channel/densenet169.json")
    parser.add_argument("--n_gpus", default="1")
    parser.add_argument("--batch_size", default="32")
    parser.add_argument("--training_config", default="training_configs/binary_crossentropy")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args


def main():
    args = parse_args()

    if args.idir == "None":
        data = args.data
    else:
        data = ','.join([os.path.join(args.idir, u) for u in os.listdir(args.idir)])

    # works on Longleaf
    cmd = 'sbatch --partition=volta-gpu  --gres=gpu:{7} --time=2-00:00:00 --qos=gpu_access src/SLURM/run_training.sh {0} {1} {2} {3} {4} {5} {6} {7}'

    cmd_ = cmd.format(args.architecture, data, args.odir, args.tag, args.batch_size, args.indices, args.training_config, args.n_gpus)

    print(cmd_)
    os.system(cmd_)


if __name__ == '__main__':
    main()