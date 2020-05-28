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

    parser.add_argument("--architecture", default = "architectures/var_size_two_channel/densenet169.json")
    parser.add_argument("--n_gpus", default = "2")
    parser.add_argument("--batch_size", default = "96")
    parser.add_argument("--training_config", default = "training_configs/mixed")

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
    cmd = 'sbatch --partition=volta-gpu  --gres=gpu:{6} --time=2-00:00:00 --qos=gpu_access src/SLURM/run_training.sh {4} {0} {1} {2} {5} {3} {7} {6}'

    ifiles = os.listdir(args.idir)

    for ifile in ifiles:
        cmd_ = cmd.format(os.path.join(args.idir, ifile), args.odir, ifile.split('.')[0], args.indices, args.architecture, args.batch_size, args.n_gpus, args.training_config)
        os.system(cmd_)

if __name__ == '__main__':
    main()
