# Supervised learning experiment on HPC

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
