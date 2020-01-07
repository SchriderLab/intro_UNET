import os
import logging, argparse
import itertools

import platform

batch_sizes = [24, 48, 96, 144, 192]
losses = ['mixed']

config_dir = 'training_configs'

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--data", default = "None")
    parser.add_argument("--indices", default = "None")
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--tag", default = "None")

    parser.add_argument("--batch_size", default = "144")

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
    if platform.platform() == 'Linux-3.10.0-1062.el7.x86_64-x86_64-with-redhat-7.7-Maipo':
        cmd = 'sbatch --partition=volta-gpu  --gres=gpu:{7} --time=2-00:00:00 --qos=gpu_access src/SLURM/run_training.sh {0} {1} {2} {3} {4} {5} {6} {7}'
    elif platform.platform() == 'Linux-3.10.0-957.27.2.el7.x86_64-x86_64-with-centos-7.6.1810-Core':
        cmd = 'sbatch --partition=GPU --gres=gpu:p100:2 --time=2-00:00:00 src/SLURM/run_training.sh {0} {1} {2} {3} {4} {5} {6} {7}'
    else:
        cmd = 'sbatch --partition=volta-gpu  --gres=gpu:{7} --time=2-00:00:00 --qos=gpu_access src/SLURM/run_training.sh {0} {1} {2} {3} {4} {5} {6} {7}'

    models = os.listdir(args.idir)

    todo = list(itertools.product(batch_sizes, losses, [os.path.join(args.idir, u) for u in models]))

    for bs, loss, model in todo:

        tag = args.tag + '_{0}_{1}_{2}'.format(bs, loss, model.split('/')[-1].split('.')[0])

        cmd_ = cmd.format(model, args.data, args.odir, tag, bs, args.indices, os.path.join(config_dir, loss), args.n_gpus)
        print(cmd_)

        os.system(cmd_)

if __name__ == '__main__':
    main()



