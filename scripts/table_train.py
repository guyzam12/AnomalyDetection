"""
Train a diffusion model on tables.
"""
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
# from scripts_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     args_to_dict,
#     add_dict_to_argparser,
# )

def main():
    args = create_argparser().parse_args()
    #dist_util.setup_dist() #TODO
    logger.configure()
    model_defaults_keys = model_defaults().keys()
    diffusion_defaults_keys = diffusion_defaults().keys()
    model_obj = model.create_model(**args_to_dict(args,model_defaults_keys))
    diffusion_obj = diffusion.create_diffusion(**args_to_dict(args,diffusion_defaults_keys))
    logger.log("creating data loader...")
    data = load_data(
        data_file=args.data_file,
        batch_size=args.batch_size,
    )
    logger.log("training...")
    train_obj = train_util.TrainLoop(model=model_obj,diffusion=diffusion_obj,data=data,batch_size=args.batch_size,lr=0.001)
    train_obj.run_loop()
    #train_obj.loss_figure()
    #model = create_model(arg,model_and_diffusion_defaults())
    #model, diffusion = create_model_and_diffusion(
    #    **args_to_dict(args, model_and_diffusion_defaults().keys())



if __name__ == "__main__":
    main()
