import os
import logging, argparse

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--idir", default = "data/")
    parser.add_argument("--model", default = "deepintraSV_v0.1.json")
    parser.add_argument("--gen_size", default = "64")
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

    return args

def main():
    args = parse_args()

    data_files = os.listdir(args.idir)

    cmd = "python3 src/models/train.py --model {0} --data {1} --gen_size {2} --odir {3} --tag {4}"

    for data_file in data_files:
        os.system(cmd.format(args.model, os.path.join(args.idir, data_file), args.gen_size, args.odir, data_file.split('.')[0]))

if __name__ == '__main__':
    main()