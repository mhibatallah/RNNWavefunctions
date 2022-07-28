In this folder, you will find an implementation of the two dimensional tensorized RNN for the square and the triangular lattices with open boundary conditions. This code was used to produce the results of our NeurIPS 2021 paper: https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_92.pdf

Our code supports multiples GPUs for speed up purposes and can also handle group symmetries on the square and the triangular lattices.

If you would like to try the 2D Heisenberg on the square lattice for 4x4 system size, you can set `J2 = 0.0` and `L = 4` using the following:

```
python Run_RNNWF_2DHeis.py --J2 0.0 --L 4 
```

To use the 4x4 RNN model to pretrain the optimization for the 6x6 system size, you can just run:

```
python Run_RNNWF_2DHeis.py --J2 0.0 --L 6
```

If you would like to run the code for the 2D Heisenberg on the triangular lattice for the same size, you can set `J2 = 1.0` as follows:

```
python Run_RNNWF_2DHeis.py --J2 1.0 --L 4
```

To be able to run the 2D Heisenberg on the square lattice with `C4v` symmetry, `A1` group character, `U(1)` symmetry (zero magnetization), you can do:

```
python Run_RNNWF_2DHeis.py --J2 0.0 --L 4 --mag_fixed True --Sz 0 --RNN_symmetry c4vsym --group_character A1
```

By default no symmetry is added on top of the RNN. We can also do the same on the triangular lattice with `C2d` symmetry:

```
python Run_RNNWF_2DHeis.py --J2 0.0 --L 4 --mag_fixed True --Sz 0 --RNN_symmetry c2dsym --group_character A1
```

In case you would like to run annealing on the triangular lattice with an initial pseudo-temperature `T0 = 0.25`, you can run the following:

```
python Run_RNNWF_2DHeis.py --J2 1.0 --L 4 --mag_fixed True --Sz 0 --T0 0.25 --Nannealing 1000 --Nwarmup 1000 --Nconvergence 1000
```

`Nannealing`, `Nwarmup`, `Nconvergence` are hyperparameters described in [our manuscript](https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_92.pdf).

Other hyperparameters can be explored in `Run_RNNWF_2DHeis.py'.

We note that in this code we use the `tensordot2` operation from the [TensorNetwork package](https://github.com/google/TensorNetwork) to speed up tensorized operations.

