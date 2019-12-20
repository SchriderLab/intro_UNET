import os
import h5py

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--ofile", default = "None")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    keys_to_write = ['x_0', 'y', 'features', 'positions', 'params']

    ifiles = sorted(os.listdir(args.idir))
    ofile = h5py.File(args.ofile, 'w')

    counter = 0

    for ifile in ifiles:
        print(counter)

        ifile = h5py.File(os.path.join(args.idir, ifile), 'r')

        for key in keys_to_write:
            ofile.create_dataset('{0}/{1}'.format(counter, key), data = np.array(ifile['0/{0}'.format(key)]))

        counter += 1

    ofile.close()



