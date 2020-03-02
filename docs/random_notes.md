We try various architectures on a bidirectional gene flow dataset.  Here is how it is run:

```python3 src/SLURM/evaluate_network.py --data data/data_128_10e5/data_bi_NN.hdf5 --model SegNet_v0.1.json --indices indices/128/128_10e5_bi.pkl --odir training_output/SegNet_v0.1 --tag SegNet```

```python3 src/SLURM/evaluate_network.py --data data/data_128_10e5/data_bi_NN.hdf5 --model deepintraSV_k0_64_128.json --indices indices/128/128_10e5_bi.pkl --odir training_output/deepIntra_k0_64 --tag deepIntra```

```python3 src/SLURM/evaluate_network.py --data data/data_128_10e5/data_bi_NN.hdf5 --model Unet++_vgg16.json --indices indices/128/128_10e5_bi.pkl --odir training_output/Unet++_vgg16 --tag Unet++_vgg16```

```python3 src/SLURM/evaluate_network.py --data data/data_128_10e5/data_bi_NN.hdf5 --model Unet++_vgg19.json --indices indices/128/128_10e5_bi.pkl --odir training_output/Unet++_vgg19 --tag Unet++_vgg19```

```python3 src/SLURM/evaluate_network.py --data data/data_128_10e5/data_bi_NN.hdf5 --model Unet++_densenet121.json --indices indices/128/128_10e5_bi.pkl --odir training_output/Unet++_densenet121 --tag Unet++_densenet121```

```python3 src/SLURM/evaluate_network.py --data data/data_128_10e5/data_bi_NN.hdf5 --model Unet++_densenet169.json --indices indices/128/128_10e5_bi.pkl --odir training_output/Unet++_densenet169 --tag Unet++_densenet169```

```python3 src/SLURM/evaluate_network.py --data data/data_128_10e5/data_bi_NN.hdf5 --model Unet++_densenet201.json --indices indices/128/128_10e5_bi.pkl --odir training_output/Unet++_densenet201 --tag Unet++_densenet201```

Command for formatting ghost data on Dogwood:

