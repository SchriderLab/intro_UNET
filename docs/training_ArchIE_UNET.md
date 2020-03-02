Training a UNET for reference-free inference of introgression from a ghost population
====================================

First we need a training set.  We'll sample a number of predictor and target images from the sliding windows
we've gathered:

```
python3 src/data/sample_archie_windows.py --ofile archie_200_10e5.hdf5 --n_samples 100000
```

Then we can train on the cluster:

```
sbatch --partition=GPU --gres=gpu:p100:2 --time=2-00:00:00 src/SLURM/run_training.sh architectures/var_size/densenet169.json /pylon5/mc5phjp/kilgore/archie_data/archie_200_10e5.hdf5 /pylon5/mc5phjp/kilgore/ds_ArchIE_experiments 10e5 32 None training_configs/mixed 2
```

Or locally if we have enough VRAM:

```
python3 src/models/train.py --model architectures/var_size/densenet169.json --data archie_200_10e5.hdf5 --tag test_v1 --gen_size 32 --odir training_results --train_config training_configs/mixed --n_gpus 1
```

Then we can evaluate our model on an independent test set (this will plot an ROC curve and compute AUROC for comparison with the
result reported in the ArchIE manuscript):

```
python3 src/models/predict_archie_UNET.py --data archie_200_data_test.hdf5 --weights weights/10e5.singleGPU.weights
```