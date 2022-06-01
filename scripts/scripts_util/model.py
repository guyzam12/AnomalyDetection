import torch as th
import torch.nn as nn
import numpy as np
import math

def create_model(
        input_size_x=4,
        input_size_emb=32,
        hidden1_size_x=64,
        hidden1_size_emb=64,
        res_input_size=64,
        res_output_size=64,
        output_size=4,
):
    return SimpleNet(input_size_x,hidden1_size_x,output_size,input_size_emb,hidden1_size_emb,res_input_size,res_output_size)


class SimpleNet(nn.Module):

    def __init__(self,input_size_x,hidden1_size_x,output_size,input_size_emb,hidden1_size_emb,res_input_size,res_output_size):
        super().__init__()
        self.dtype = th.float32

        self.input_blocks_x = nn.Linear(input_size_x,hidden1_size_x)
        self.input_blocks_emb = nn.Linear(input_size_emb,hidden1_size_emb)
        self.res_block = ResBlock(res_input_size,res_output_size)
        self.output_blocks = nn.Linear(res_output_size,output_size)
        self.relu = nn.ReLU()

    # def forward(self, x):
    #     out = self.fc1(x)
    #     #out = self.bn1(out)
    #     out = self.relu1(out)
    #     out = self.fc2(out)
    #     out = self.relu2(out)
    #     out = self.fc3(out)
    #     out = self.relu3(out)
    #     out = self.fc4(out)
    #     return out

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x ...] Tensor of outputs.
        """
        hs = []
        emb = timestep_embedding(timesteps, 32)
        x_out = self.input_blocks_x(x.float())
        emb_out = self.input_blocks_emb(emb)
        out = self.res_block(x_out,emb_out)
        out = self.relu(out)
        out = self.output_blocks(out)
        return out


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            input_size,
            output_size,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_size, output_size)
        )

    def forward(self, x, emb):
        h = x + emb
        h = self.layers(h)
        return x + h

