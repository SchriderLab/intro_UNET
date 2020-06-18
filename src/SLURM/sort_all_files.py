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

    parser.add_argument("--format_config", default = "None")

    parser.add_argument("--archie", action = "store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.odir):
        os.mkdir(args.odir)
        logging.debug('root: made output directory {0}'.format(args.odir))

    return args

def main():
    args = parse_args()

    cmd = 'sbatch -n 24 -t 2-00:00:00 --wrap "mpirun -oversubscribe python3 src/data/sort_training_data.py --ifile {0} --ofile {1} --format_config {2} --two_channel --verbose"'

    if args.archie:
        cmd = cmd + ' --y_channel 1"'
    else:
        cmd = cmd + '"'

    ifiles = sorted(os.listdir(args.idir))

    for ifile in ifiles:
        cmd_ = cmd.format(os.path.join(args.idir, ifile), os.path.join(args.odir, ifile), args.format_config)
        print(cmd_)
        os.system(cmd_)

if __name__ == '__main__':
    main()