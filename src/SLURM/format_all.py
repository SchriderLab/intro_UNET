import os
import logging, argparse

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default="None")
    parser.add_argument("--odir", default="None")
    parser.add_argument("--format_mode", default = "sort_NN")

    parser.add_argument("--n_individuals", default = "128")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    directories = os.listdir(args.idir)
    directories = [u for u in directories if os.path.isdir(os.path.join(args.idir, u))]

    cmd = 'sbatch -t 1-00:00:00 --wrap "python3 src/data/format_data.py --idir {0} --format_mode {1} --n_individuals {2} --ofile {3} --verbose"'

    for directory in directories:
        tag = directory

        cmd_ = cmd.format(os.path.join(args.idir, directory), args.format_mode, args.n_individuals, os.path.join(args.odir, '{0}.hdf5'.format(tag)))

        os.system(cmd_)
        print(cmd_)

if __name__ == '__main__':
    main()

# no_sorting/bi
# max_match_sorted/bi

# python3 src/SLURM/evaluate_network.py --idir architectures/var_size/ --data /proj/dschridelab/ddray/intro_data/ArchIE_data/max_match.hdf5 --indices archie_64_indices.pkl --odir /pine/scr/d/d/ddray/ArchIE_grids/max_match --tag ArchIE