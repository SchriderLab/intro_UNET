import os
import itertools

ST = [1, 4, 16]
MT = [0.1, 0.25, 0.75]
MP = [0.1, 0.2, 0.25]

cmd = 'python3 src/data/generate_data.py --odir /proj/dschridelab/introgression_data/sims_perturbed_ghost/{0}_{1}_{2}_128 --st {0} --mt {1} --mp {2} --n_per_pop 200 --n_jobs 12500'

possible = itertools.product(ST, MT, donor_pops)

for i, j, k in possible:
    os.system(cmd.format(i, j, k))