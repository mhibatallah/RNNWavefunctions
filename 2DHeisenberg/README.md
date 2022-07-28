In this folder, you will find an implementation of the two dimensional tensorized RNN for the square and the triangular lattices.

If you would like to try the 2D Heisenberg on the square lattice, you can set `J2 = 0.0`.

```
python Run_RNNWF_2DHeis.py --J2 0.0 --L 6 --mag_fixed True --Sz 0 --RNN_symmetry c4vsym --group_character A1 --numunits 50 --numsamples 100 --numsteps 10000 
```

If you would like to run the code for the 2D Heisenberg on the triangular lattice, you can set `J2 = 1.0`.

```
python Run_RNNWF_2DHeis.py --J2 1.0 --L 6 --mag_fixed True --Sz 0 --RNN_symmetry c4vsym --group_character A1 --numunits 50 --numsamples 100 --numsteps 10000 
```

We note that in this code we use the `tensordot2` operation from the [TensorNetwork package](https://github.com/google/TensorNetwork) to speed up tensorized operations.

