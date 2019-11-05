Deep Neural Nets for segmentation of introgressed regions of the genome
==============================

Package for training and testing of neural nets on data generated with SLiM.  

How to generate data (done inside a SLURM cluster):

```python3 src/data/generate_data.py --odir /somewhere/for/the/raw/sims --n_jobs 1000 --n_replicates 100000```

This will generate 10000 replicates spread over 1000 jobs.  You should then gzip the results for the next step:

```
cd /somewhere/for/the/raw/sims 
gzip *.ms *.log
````

Format the data into an HDF5 file:

```python3 src/data/format_data.py --idir /somewhere/for/the/raw/sims --ofile data.hdf5```

Train:

```python3 src/models/train.py --model deepintraSV_v0.1.json --data data.hdf5 --odir training_output/ --tag test_training```

Check out the arguments for train.py.  You can control the amount of data used for validation and testing (by default both are 10 %), which model you use for training (as a .JSON file), the batch size, the number of gpus, etc.

The training script will save the weights, training history, and the test result (loss and accuracy of the best network on the test set) in the output directory specified named with the 'tag' specified.
