"""
Train a diffusion model on tables.
"""

import torch as th
import argparse

from scripts_util.argparser_util import (
    create_argparser,
    model_defaults,
    diffusion_defaults,
    args_to_dict,
    update_args
    )
from scripts_util import (
    logger,
    model,
    diffusion,
    train_util
    )
from scripts_util.table_dataset import (
    load_data,
)


def main():
    defaults = dict(
        data_file="",
        output_model_name="",
        batch_size=16,
        row_size=0,
        log_interval=200,
        save_interval=50000,
        lr=0.0001,
        lr_anneal_steps=500000,
        load_model="",
        project_path="/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion"
    )

    # Create argparser with default parameters
    args = create_argparser(defaults).parse_args()
    args = update_args(args, True, False)

    # GPU Handling
    #dist_util.setup_dist() #TODO

    # Logger configuration
    logger.configure()

    # Creation of model and diffusion models
    logger.log("creating model and diffusion...")
    model_obj = model.create_model(**args_to_dict(args,model_defaults().keys()))
    if args.load_model != "":
        model_obj.load_state_dict(th.load(args.load_model))
    diffusion_obj = diffusion.create_diffusion(**args_to_dict(args,diffusion_defaults().keys()))

    # Creating of data loader
    logger.log("creating data loader...")
    data = load_data(
        data_file=args.data_file,
        batch_size=args.batch_size,
        output_model_name=args.output_model_name,
    )

    # Training
    logger.log("training...")
    args_defaults_keys = args_to_dict(args,args.__dict__).keys()
    train_obj = train_util.TrainLoop(**args_to_dict(args,args_defaults_keys),model=model_obj,diffusion=diffusion_obj,data=data)
    train_obj.run_loop()

    # Save model
    #th.save(model_obj.state_dict(),args.output_model_name)

    #train_obj.acc_figure()
    #train_obj.loss_figure()



if __name__ == "__main__":
    main()
