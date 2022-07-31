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
        data_file="/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/datasets/gaussian.csv",
        batch_size=1,
        row_size=1,
        log_interval=100,
        save_interval=1000,
        lr=0.0001,
        lr_anneal_steps=10000,
    )
    # Create argparser with default parameters
    args = create_argparser(defaults).parse_args()
    #dist_util.setup_dist() #TODO
    logger.configure()
    logger.log("creating model and diffusion...")
    model_obj = model.create_model(**args_to_dict(args,model_defaults().keys()))
    diffusion_obj = diffusion.create_diffusion(**args_to_dict(args,diffusion_defaults().keys()))
    logger.log("creating data loader...")
    data = load_data(
        data_file=args.data_file,
        batch_size=args.batch_size,
    )
    logger.log("training...")
    args_defaults_keys = args_to_dict(args,args.__dict__).keys()
    train_obj = train_util.TrainLoop(**args_to_dict(args,args_defaults_keys),model=model_obj,diffusion=diffusion_obj,data=data)
    train_obj.run_loop()
    th.save(model_obj.state_dict(),"/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/models/gaussian_v2.pt")
    #train_obj.acc_figure()
    #train_obj.loss_figure()



if __name__ == "__main__":
    main()
