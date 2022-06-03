"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
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
        model_path="/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/models/temp.pt",
        data_file="/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/datasets/Iris/Iris_one_row.csv",
        batch_size=1,
        log_interval=100,
        save_interval=10000,
        lr=0.0001,
        lr_anneal_steps=5000,
        num_samples=10,
    )
    # Create argparser with default parameters
    args = create_argparser(defaults).parse_args()
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
    while len(all_rows) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_fn = (
            diffusion_obj.p_sample_loop
        )
        sample = sample_fn(
            model_obj,
            (args.batch_size, 10),
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )

        all_rows.append(sample)
        logger.log(f"created {len(all_rows) * args.batch_size} samples")

    print("hi")
    '''
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")
    print("hi")
    #diffusion_obj = diffusion.create_diffusion(**args_to_dict(args, diffusion_defaults().keys()))
    #model_obj.eval()
    #logger.log("sampling...")
    #all_images = []
    #all_labels = []
    '''
if __name__ == "__main__":
    main()

