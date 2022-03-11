# Standard library imports
import argparse
import random
import os
import sys
import json

#Third party libraries
import torch
import numpy as np


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file, 'r') as json_file:
            return json.load(json_file)
    else:
        return config_file


def parse_arguments_generate_dataset():
    ap = argparse.ArgumentParser(description='Using of media pipe in PSL')
    
    ap.add_argument('--inputPath', type=str, default="/data/shuaman/psl_gan/Data/Videos/Segmented_gestures/",
                    help='relative path of images input.')

    ap.add_argument('-w', '--withLineFeature', default=False, action='store_true',
                    help="use line features")

    ap.add_argument('-l', '--leftHandLandmarks', default=False, action='store_true',
                    help='Get left hand landmarks')

    ap.add_argument('-r', '--rightHandLandmarks', default=False, action='store_true',
                    help='Get right hand landmarks')

    ap.add_argument('-f', '--faceLandmarks', default=False, action='store_true',
                    help='Get face landmarks')

    ap.add_argument('-m', '--minframes', type=int, default=10,
                    help='Number of frames of each video')

    ap.add_argument('-i', '--mininstances', type=int, default=4,
                    help='Number of instances of each class')

    ap.add_argument('-a', '--addExtraJoint', default=False, action='store_true',
                    help='Add the joint number 33 (middle point of 12 and 11)')

    ap.add_argument('-p', '--porcFrameComplet', type=float, default=0.2,
                    help='Min percentage to fill frames')



    args = ap.parse_args()

    return args



def parse_arguments_train():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--wandb', default=False, action='store_true',
                    help="use weights and biases")
    ap.add_argument('-n  ', '--no-wandb', dest='wandb', action='store_false',
                    help="not use weights and biases")
    ap.add_argument('-a', '--run_name', required=False, type=str, default=None,
                    help="name of the execution to save in wandb")
    ap.add_argument('-t', '--run_notes', required=False, type=str, default=None,
                    help="notes of the execution to save in wandb")

    args = ap.parse_args()

    return args



def configure_model(config_file, use_wandb):
    config_file = parse_configuration(config_file)
    config = dict(
        n_epochs = config_file["hparams"]["n_epochs"],
        batch_size = config_file["hparams"]["batch_size"],
        lr = config_file["hparams"]["lr"],
        latent_dim = config_file["hparams"]["latent_dim"],
        mlp_dim = config_file["hparams"]["mlp_dim"],
        lambda_gp = config_file["hparams"]["lambda_gp"],
        n_critic = config_file["hparams"]["n_critic"],
        n_samples = config_file["hparams"]["n_samples"],
        n_samples_plot = config_file["hparams"]["n_samples_plot"],

        matrix_data = config_file["datasets"]["matrix_data"],
        signs_to_use = config_file["datasets"]["signs_to_use"],

        save_weights = config_file["save_weights"],
        num_backups = config_file["num_backups"],
        path_saved_weights = config_file["path_saved_weights"],
        weights_default = config_file["weights_default"]
    )

    if not use_wandb:
        config = type("configuration", (object,), config)

    return config