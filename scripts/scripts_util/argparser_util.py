import argparse

def create_argparser(defaults):
    """
    High-level definitions
    """
    defaults.update(model_defaults())
    defaults.update(diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def model_defaults():
    """
    Defaults for model.
    """
    mod_defaults = dict( ##TODO: Complete
        row_size=1,
        emb_size=32,
        hidden1_size_x=64,
        hidden1_size_emb=64,
        res_input_size=64,
        res_output_size=64,
    )
    return mod_defaults


def diffusion_defaults():
    """
    Defaults for diffusion.
    """
    diff_defaults = dict( ##TODO: Complete
        steps=1000,
    )
    return diff_defaults


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}
