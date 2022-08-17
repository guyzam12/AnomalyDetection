import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.optim import AdamW

from . import logger
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import re

INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        log_interval,
        lr_anneal_steps,
        save_interval,
        output_model_name,
        data_file,
        **kwargs,
    ):
        self.data_file = pd.read_csv(data_file,header=None)
        #self.data_file = self.data_file[self.data_file.columns[1:]]
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.lr_anneal_steps = lr_anneal_steps
        self.opt = AdamW(self.model.parameters(), weight_decay=0.001)
        self.criterion = th.nn.MSELoss()
        self.step = 0
        self.loss = 0
        self.loss_acc = th.tensor([])
        self.steps_acc = th.tensor([])
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.weights = np.ones(diffusion.num_timesteps)
        self.output_model_name = output_model_name


    def run_loop(self):
        while (
            self.step <= self.lr_anneal_steps
        ):
            batch = next(self.data)
            self.run_step(batch)
            self.step += 1

    def loss_figure(self):
        x,y = self.steps_acc.detach().numpy(),self.loss_acc.detach().numpy()
        plt.plot(x,y)
        plt.show()

    def acc_figure(self):
        total_samples = 0
        acc = 0
        acc_list = []
        total_same_outputs = 0
        for i in range(100):
            batch,labels = next(self.data)
            outputs = self.model(batch.float())
            labels_argmax, outputs_argmax = th.argmax(labels,dim=1),th.argmax(outputs,dim=1)
            same_outputs = th.sum(labels_argmax == outputs_argmax)
            total_same_outputs += same_outputs
            total_samples += self.batch_size
            acc_list.append(total_same_outputs/total_samples)
        plt.plot(np.arange(100),acc_list)
        plt.show()


    def run_step(self, batch):
        self.forward_backward(batch)
        #self.opt.zero_grad()
        # outputs = self.model(batch.float())
        # #self.loss = self.criterion(outputs, labels.float())
        # self.loss.backward()
        self.opt.step()
        if self.step % self.log_interval == 0:
            self.log_step()
            self.loss_acc = th.cat((self.loss_acc,th.unsqueeze(self.loss, dim=0)),dim=0)
            self.steps_acc = th.cat((self.steps_acc,th.unsqueeze(th.tensor(self.step),dim=0)),dim=0)
            logger.dumpkvs()
        if self.step % self.save_interval == 0 and self.step != 0:
            if "_" in self.output_model_name:
                self.output_model_name = re.sub(r"_.*.pt","_"+str(int(self.step/1000))+"kstep.pt",self.output_model_name)
            else:
                self.output_model_name = re.sub(r".pt","_"+str(int(self.step/1000))+"kstep.pt",self.output_model_name)

            self.data_file.to_pickle(self.output_model_name.replace('.pt', '.pkl'))
            th.save(self.model.state_dict(), self.output_model_name)

    def forward_backward(self, batch):
        self.opt.zero_grad()
        #self.loss = self.criterion(batch, labels.float())

        for i in range(0, batch.shape[0]):
            t, weights = self.sample(batch.shape[0])
            compute_losses = functools.partial(
                 self.diffusion.training_losses,
                 self.model,
                 batch,
                 t,
             )
            losses = compute_losses()
            self.loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                 self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.loss.backward()

    def sample(self, batch_size):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float()
        return indices, weights


    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        #logger.logkv("loss", self.loss)
        #logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

