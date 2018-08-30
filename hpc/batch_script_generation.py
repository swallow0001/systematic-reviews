# Batch script generation
#
# Arguments:
#     -n: Number of batch files.
#
# Authors: Roel Brouwer, Kees van Eijden, Jonathan de Bruin
#
# Dependencies: sklearn, numpy
# License: BSD-3-Clause

# pylint: disable=C0321

import os
import argparse
from math import pow, ceil
from itertools import product
from utils import *


def make_bash_file(n_processes=24, n_batch=0):
    """Generic function to create batch files for HPC cluster."""
    parameters = read_parameters()
    
    # recusive if len(parameters > 24)
    if len(parameters) > n_processes:
        make_bash_file(parameters[n_processes:], n_processes=n_processes, n_batch=n_batch + 1)
        parameters = parameters[:n_processes]

    # generate script content
    bash_command = "#!/bin/bash\n"
    bash_command += "#SBATCH -t 00:15:00\n"
    bash_command += "#SBATCH -N 1\n"
    bash_command += "cd $HOME/systematic_review\n"
    bash_command += "module load python/3.5.2-intel-u2\n"
    bash_command += "date\n"

    for i, parm in enumerate(parameters):
        
        i_job = int(i + n_batch * n_processes)
        bash_command += "python ./python/sr_lstm.py " \
            "-T {} -training_size {} -allowed_FN {} -init_included_papers {} -dataset {}  &\n".format(i_job, parm[0], parm[1], parm[2], parm[3])

    bash_command += "wait\n"
    bash_command += "date\n"

    # write batch commands to file
    if not os.path.exists('batch_files'): os.makedirs('batch_files')  # noqa
    export_fp = os.path.join('batch_files', "batch-{}.sh".format(n_batch))
    with open(export_fp, "w") as f:
        f.writelines(bash_command)

    return bash_command


if __name__ == '__main__':

    # hyperparameter: grid search
    sampling_size = 10
    
    training_size= list(range(50,501,50)) 
    allowed_FN = list(range(0,10))
    init_included_papers=[10]
    dataset=['ptsd']
   
    hyperparameters = list(product(training_size, allowed_FN, init_included_papers, dataset))

    training_size= list(range(50,501,50)) 
    allowed_FN = list(range(0,2))
    init_included_papers=[10,15,20]
    dataset=['ptsd']

    hyperparameters += list(product(training_size, allowed_FN, init_included_papers, dataset))
        
    ##remove duplicated combinations
    hyperparameters = list(set(hyperparameters)) * sampling_size

    names =['training_size','allowed_FN','init_included_papers','dataset']
    save_parameters(names,hyperparameters)

    # parse the arguments
    parser = argparse.ArgumentParser(description='sr options')
    parser.add_argument("-n", default=1, type=int, help='n files.')
    sr_args = parser.parse_args()

    # the number of batch files to generate
    n_processes = int(ceil(len(hyperparameters) / sr_args.n))

    make_bash_file(n_processes=n_processes)
