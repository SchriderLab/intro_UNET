import os
import numpy as np
import logging, argparse

def parse_args():
    # Argument Parser    
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--odir", default = "/proj/dschridelab/introgression_data/")
    parser.add_argument("--n_jobs", default = "1000")
    parser.add_argument("--n_replicates", default = "200000")

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

    n_jobs = int(args.n_jobs)
    n_replicates = int(args.n_replicates)

    replicates_per = n_replicates // n_jobs

    # should have an even number of pop1 to pop2 vs. pop2 to pop1
    # eventually we'll put in introgression both ways
    cmd = 'sbatch -t 2-00:00:00 --wrap "python3 runAndParseSlim.py introg.slim {0} 3000 {1} {2} 1> {3}"'

    counter = 0

    # 1 to 2
    for ix in range(n_jobs // 2):
        os.system(cmd.format(replicates_per, 1, os.path.join(args.odir, 'sim.{0:06d}.log'.format(counter)), os.path.join(args.odir, 'sim.{0:06d}.ms'.format(counter))))
        counter += 1

    # 2 to 1
    for ix in range(n_jobs // 2):
        os.system(cmd.format(replicates_per, 2, os.path.join(args.odir, 'sim.{0:06d}.log'.format(counter)), os.path.join(args.odir, 'sim.{0:06d}.ms'.format(counter))))
        counter += 1

if __name__ == '__main__':
    main()
    
    
