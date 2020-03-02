Solving the SNP-level introgression problem
====================================

We wish to train a FCN (fully convolutional net) to predict, given aligned SNP data from two populations, which polymorphisms are introgressed
and which are native.  First, we simulate introgression data between two populations with a variety of migration probabilities and migration times.
There are two populations and thus three distinct cases of introgression we're interested in: migration from population A to B, B to A, and the bidirectional
case.  To simulate:

```
python3 src/data/generate_data.py --odir /proj/dschridelab/introgression_data/sims_128_v2/10e5/sims_raw_AB --donor_pop 1 --n_per_pop 64
python3 src/data/generate_data.py --odir /proj/dschridelab/introgression_data/sims_128_v2/10e5/sims_raw_BA --donor_pop 2 --n_per_pop 64
python3 src/data/generate_data.py --odir /proj/dschridelab/introgression_data/sims_128_v2/10e5/sims_raw_bi --donor_pop 3 --n_per_pop 64

sbatch --wrap "gzip /proj/dschridelab/introgression_data/sims_128_v2/10e5/sims_raw_AB/*.ms /proj/dschridelab/introgression_data/sims_128_v2/10e5/sims_raw_AB/*.log"
sbatch --wrap "gzip /proj/dschridelab/introgression_data/sims_128_v2/10e5/sims_raw_BA/*.ms /proj/dschridelab/introgression_data/sims_128_v2/10e5/sims_raw_BA/*.log"
sbatch --wrap "gzip /proj/dschridelab/introgression_data/sims_128_v2/10e5/sims_raw_bi/*.ms /proj/dschridelab/introgression_data/sims_128_v2/10e5/sims_raw_bi/*.log"
```

This generates and gunzips 100000 replicates for each case.  The data must then be formatted such that it works with the training routines developed.
The formatting routine can also sort the matrix according to some algorithm (see the documentation for all options available)

```
sbatch -t 1-00:00:00 --wrap "python3 src/data/format_data.py --idir /proj/dschridelab/introgression_data/sims_128_v2/10e5/sims_raw_AB --ofile /proj/dschridelab/ddray/data_AB_NN.hdf5 --format_mode sort_NN --n_individuals 128"
sbatch -t 1-00:00:00 --wrap "python3 src/data/format_data.py --idir /proj/dschridelab/introgression_data/sims_128_v2/10e5/sims_raw_BA --ofile /proj/dschridelab/ddray/data_BA_NN.hdf5 --format_mode sort_NN --n_individuals 128"
sbatch -t 1-00:00:00 --wrap "python3 src/data/format_data.py --idir /proj/dschridelab/introgression_data/sims_128_v2/10e5/sims_raw_bi --ofile /proj/dschridelab/ddray/data_bi_NN.hdf5 --format_mode sort_NN --n_individuals 128"
```

Then let's train on the full dataset over a number of architectures and batch sizes to decide on a model:

```
python3 src/SLURM/evaluate_network.py --idir architectures/var_size/ --data /proj/dschridelab/ddray/intro_data/data_128_10e5/data_AB_NN.hdf5,/proj/dschridelab/ddray/intro_data/data_128_10e5/data_BA_NN.hdf5,/proj/dschridelab/ddray/intro_data/data_128_10e5/data_bi_NN.hdf5 --indices indices/128/128_10e5_all.pkl --odir /pine/scr/d/d/ddray/128_all_architectures/ --tag sweep
```

The model and batch size that minimize loss are the UNet++ with a DenseNet201 backbone and 96 respectively.

