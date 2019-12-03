import os
import logging, argparse
import itertools

batch_sizes = [128]
losses = ['mixed']

config_dir = 'training_configs'

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--model", default = "None")
    parser.add_argument("--data", default = "None")
    parser.add_argument("--indices", default = "None")
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--tag", default = "None")

    parser.add_argument("--cluster", default = "longleaf")

    parser.add_argument("--n_gpus", default = "2")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    # model, data, output directory, tag, batch size, indices
    if args.cluster == 'longleaf':
        cmd = 'sbatch --partition=volta-gpu  --gres=gpu:{7} --time=2-00:00:00 --qos=gpu_access src/SLURM/run_training.sh {0} {1} {2} {3} {4} {5} {6} {7}'

    todo = list(itertools.product(batch_sizes, losses))

    for bs, loss in todo:

        tag = args.tag + '_{0}_{1}'.format(bs, loss)

        cmd_ = cmd.format(args.model, args.data, args.odir, tag, bs, args.indices, os.path.join(config_dir, loss), args.n_gpus)
        print(cmd_)

        os.system(cmd_)

if __name__ == '__main__':
    main()



