import os
import logging, argparse
import itertools

import platform

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--idir_AB", default = "None")
    parser.add_argument("--idir_BA", default = "None")

    parser.add_argument("--architecture", default = "architectures/var_size_2ch_to_2ch/densenet169.json")
    
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
    cmd = 'sbatch --partition=volta-gpu  --gres=gpu:2 --time=2-00:00:00 --qos=gpu_access src/SLURM/run_training.sh {4} {0} {1} {2} 32 {3} training_configs/binary_crossentropy 2'

    ifiles = os.listdir(args.idir_AB)

    for ifile in ifiles:
        cmd_ = cmd.format(os.path.join(args.idir_AB, ifile) + "," + os.path.join(args.idir_BA, ifile), args.odir, ifile.split('.')[0], args.indices, args.architecture)
        os.system(cmd_)

if __name__ == '__main__':
    main()
