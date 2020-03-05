import os
import itertools

ST = [0.6, 1.2, 2.4]
MT = [0.1, 0.2, 0.25]
MP = [0.005, 0.02, 0.08]

cmd = 'python3 src/data/generate_data_ghost.py --odir /proj/dschridelab/introgression_data/sims_perturbed_ghost/{0}_{1}_{2}_200 --st {0} --mt {1} --mp {2} --n_per_pop 200 --n_jobs 1250'

possible = list(itertools.product(ST, MT, MP))

print(len(possible))

for i, j, k in possible:
    os.system(cmd.format(i, j, k))