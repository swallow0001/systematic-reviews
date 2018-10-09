# Supervised learning experiment on HPC


## STEP X: prepare datasets
## STEP 1: generate batch files

Generate the batch files for active learning, use the following command:

``` bash
Rscript make_sr_lstm_batch.R [DATASET_NAME] --active
```

and for passive learning the following command:

``` bash
Rscript make_sr_lstm_batch.R [DATASET_NAME] --no-active
```

Working example: 

``` bash
Rscript make_sr_lstm_batch.R ptsd --active
```

To speed up the computations on the HPC, several Python objects are generated
beforehand and stored in a pickle file. This file makes it possible to load
the objects really fast on each core on the HPC cluster.

Create a pickle file with the data, labels and embedding layer with the
following shell command:  

``` bash
python hpc/data_prep.py --dataset=ptsd
```


## WIP

This subfolder contains an experiment to run a supervised learning simulation
on the SurfSara HPC infrastructure.

Generate the batch files with the following line of command line code.
```
python batch_script_generation.py
```

A single run of the simulation is executed with the following line of code:
```
python sr_lstm.py --training_size=500
```

