"""
Train a diffusion model on tables.
"""

import torch as th
import argparse
import pandas as pd

from scripts_util.argparser_util import (
    create_argparser,
    model_defaults,
    diffusion_defaults,
    args_to_dict,
    update_args_anomaly
    )
from scripts_util import (
    logger,
    model,
    diffusion,
    train_util_anomaly2,
    )
from scripts_util.table_dataset_anomaly import (
    load_data,
    TableDatasetAnomaly
)


def main():
    defaults = dict(
        data_file="",
        train_data_file="",
        output_anomaly_name="",
        batch_size=0,
        row_size=0,
        log_interval=200,
        save_interval=200,
        lr=0.0001,
        lr_anneal_steps=500000,
        load_model="",
        project_path="/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion"
    )

    # Argparser with default parameters
    args = create_argparser(defaults).parse_args()
    args = update_args_anomaly(args)

    # Logger configuration
    logger.configure()

    # Data object
    data_obj = TableDatasetAnomaly(
        args.data_file,
        args.train_data_file,
    )
    #  Data Loader
    logger.log("creating data loader...")
    data = load_data(
        data_obj=data_obj,
        batch_size=args.batch_size,
    )

    args.row_size = data_obj.get_row_size() # Update args row size

    # GPU Handling
    #dist_util.setup_dist() #TODO


    # Creation of model and diffusion models
    logger.log("creating model and diffusion...")
    model_obj = model.create_model(**args_to_dict(args,model_defaults().keys()))
    diffusion_obj = diffusion.create_diffusion(**args_to_dict(args,diffusion_defaults().keys()))
    model_obj.load_state_dict(th.load(args.load_model))
    model_obj.eval()
    logger.log("Calculating anomaly scores...")
    number_of_rows = data_obj.get_num_of_rows()
    args_defaults_keys = args_to_dict(args,args.__dict__).keys()
    train_obj = train_util_anomaly2.AnomalyLoop(**args_to_dict(args, args_defaults_keys), model=model_obj,
                                             diffusion=diffusion_obj, data=data, data_obj=data_obj)
    #while step < number_of_rows:
    train_obj.run_loop()
        #batch, idx = next(data)
        #step += 1

    #args_defaults_keys = args_to_dict(args,args.__dict__).keys()
    #train_obj = train_util_anomaly.TrainLoop(**args_to_dict(args,args_defaults_keys),model=model_obj,diffusion=diffusion_obj,data=data,data_obj=data_obj)
    #train_obj.run_loop()

if __name__ == "__main__":
    main()
