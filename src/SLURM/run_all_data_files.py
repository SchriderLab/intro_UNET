import os
import logging, argparse
import itertools

import platform

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--indices", default = "None")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    # data, odir, tag, indices
    cmd = 'sbatch --partition=volta-gpu  --gres=gpu:2 --time=2-00:00:00 --qos=gpu_access src/SLURM/run_training.sh architectures/var_size_two_channel/densenet169.json {0} {1} {2} 96 {3} training_configs/mixed 2'

    ifiles = os.listdir(args.idir)

    for ifile in ifiles:
        cmd_ = cmd.format(os.path.join(args.idir, ifile), args.odir, ifile.split('.')[0], args.indices)
        os.system(cmd_)

if __name__ == '__main__':
    main()