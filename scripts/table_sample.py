"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import torch as th
import argparse
import numpy as np
import pandas as pd
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
        model_path="",
        denorm_file="",
        batch_size=1,
        row_size=1,
        log_interval=100,
        save_interval=10000,
        lr=0.0001,
        lr_anneal_steps=5000,
        num_samples=1000,
    )
    # Create argparser with default parameters
    args = create_argparser(defaults).parse_args()
    args.output_samples_npz = '/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/generated_samples/{0}.npz'.format(args.model_path)
    args.model_path = '/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/models/{0}.pt'.format(args.model_path)
    args.denorm_file=args.model_path.replace('.pt','.pkl')
    #defaults['denorm_file'] = defaults['model_path'].replace('.pt','.pkl')
    #dist_util.setup_dist() #TODO
    logger.configure()
    logger.log("creating model and diffusion...")
    model_obj = model.create_model(**args_to_dict(args, model_defaults().keys()))
    diffusion_obj = diffusion.create_diffusion(**args_to_dict(args,diffusion_defaults().keys()))
    model_obj.load_state_dict(th.load(args.model_path))
    model_obj.eval()
    logger.log("sampling...")
    all_rows = []
    all_labels = []
    data_file = pd.read_pickle(args.denorm_file)
    mean_per_col = th.tensor(data_file.mean())
    max_per_col = th.tensor(data_file.max())
    min_per_col = th.tensor(data_file.min())
    while len(all_rows) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_fn = (
            diffusion_obj.p_sample_loop
        )
        sample = sample_fn(
            model_obj,
            (args.batch_size, args.row_size),
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )

        sample = denorm_sample(sample,max_per_col,min_per_col)

        all_rows.append(sample.numpy())
        logger.log(f"created {len(all_rows) * args.batch_size} samples")

    np.savez(args.output_samples_npz,all_rows)


def denorm_sample(sample,max_per_col,min_per_col):
    sample = sample*(max_per_col - min_per_col)+min_per_col
    return sample

if __name__ == "__main__":
    main()

