# Standard library imports
import argparse
import os
import sys

# Third party imports
import wandb 

# Local imports
from utils.parse_arguments import parse_arguments_train, set_seed, configure_model
from models.psl_gan import PSLGAN


CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "psl_gan"
ENTITY = "stevramos"


def is_there_arg(args, master_arg):
    if(master_arg in args):
        return True
    else:
        return False


def train(config_file, use_wandb, run_name, run_notes):
    set_seed(32)
    config = configure_model(config_file, use_wandb)
    
    if use_wandb:
        wandb.init(project=PROJECT_WANDB, entity=ENTITY, config=config, name=run_name, notes=run_notes)
        config = wandb.config
        wandb.watch_called = False
    
    psl_gan = PSLGAN(config, use_wandb)
    psl_gan.train()



if __name__ == '__main__':
    use_sweep = is_there_arg(sys.argv, '--use_sweep')

    if not use_sweep:
        args = parse_arguments_train()
        use_wandb = args.wandb
        run_name = args.run_name
        run_notes = args.run_notes
    else:
        use_wandb = True
        run_name = None
        run_notes = None

    train(CONFIG_FILENAME, use_wandb=use_wandb, run_name=run_name, run_notes=run_notes)