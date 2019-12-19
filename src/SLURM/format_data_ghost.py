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



    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.odir):
        os.mkdir(args.odir)
        logging.debug('root: made output directory {0}'.format(args.odir))
    else:
        os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args


def main():
    args = parse_args()

    cmd = 'sbatch -t 1-00:00:00 --wrap "python3 src/data/format_data_ghost_longleaf.py --ms {0} --log {1} --out {2} --ofile {3}"'

    ms_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if 'ms.gz' in u])
    log_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if 'log.gz' in u])
    out_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if '.out' in u])

    for ix in range(len(ms_files)):
        ms = ms_files[ix]
        log = log_files[ix]
        out = out_files[ix]

        print(cmd.format(ms, log, out, os.path.join(args.odir, '{0:06d}.hdf5'.format(ix))))
        os.system(cmd.format(ms, log, out, os.path.join(args.odir, '{0:06d}.hdf5'.format(ix))))



if __name__ == '__main__':
    main()


