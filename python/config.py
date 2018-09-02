"""Module with parameter configurations."""

import os

##### GLOBAL VARIABLES

# project root
ROOT_DIR = ".."

##### FOLDER STRUCTURE

# log dir (for ML models)
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# output dir
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

# data dir
DATA_DIR = os.path.join(ROOT_DIR, "data")

# dataset dir
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
PTSD_DIR = os.path.join(DATASETS_DIR, "ptsd_review", "csv")
DRUG_DIR = os.path.join(DATASETS_DIR, "drug_class_review")

# glove dir
GLOVE_DIR = os.path.join(DATA_DIR, "pretrained_models", "word2vec")

##### FILE LOCATIONS
PTSD_PATH = os.path.join(PTSD_DIR, "schoot-lgmm-ptsd-traindata.csv")
