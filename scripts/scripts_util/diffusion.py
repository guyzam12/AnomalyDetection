import argparse
import inspect

from . import gaussian_diffusion as gd
#from .respace import SpacedDiffusion, space_timesteps
#from .unet import SuperResModel, UNetModel, EncoderUNetModel

NUM_CLASSES = 1000


def create_diffusion(
        steps=100,
        noise_schedule="linear",
):
    return create_gaussian_diffusion(steps,noise_schedule)


def create_gaussian_diffusion(
        steps=100,
        noise_schedule="linear",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    loss_type = gd.LossType.MSE
    return gd.GaussianDiffusion(
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=loss_type,
        rescale_timesteps=False,
    )

