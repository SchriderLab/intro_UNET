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
    parser.add_argument("--n_replicates", default = "100000")

    parser.add_argument("--n_per_pop", default = "24")

    parser.add_argument("--donor_pop", default = "1")

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
    cmd = 'sbatch -o {4} -t 2-00:00:00 --wrap "python3 src/data/runAndParseSlim.py src/data/introg_ghost_2pop_archieDemog.slim {0} 1000000 {1} {2} 1> {3} {5}"'

    counter = 0

    for ix in range(n_jobs):
        print(cmd.format(replicates_per, args.donor_pop, os.path.join(args.odir, 'sim.{0:06d}.log'.format(counter)), os.path.join(args.odir, 'sim.{0:06d}.ms'.format(counter)), os.path.join(args.odir, 'sim.{0:06d}.out'.format(counter)), args.n_per_pop))
        os.system(cmd.format(replicates_per, args.donor_pop, os.path.join(args.odir, 'sim.{0:06d}.log'.format(counter)), os.path.join(args.odir, 'sim.{0:06d}.ms'.format(counter)), os.path.join(args.odir, 'sim.{0:06d}.out'.format(counter)), args.n_per_pop))
        counter += 1

if __name__ == '__main__':
    main()