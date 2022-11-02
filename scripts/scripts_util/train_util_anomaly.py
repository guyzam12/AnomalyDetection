import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.optim import AdamW
from sklearn import metrics

from . import logger
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import re

INITIAL_LOG_LOSS_SCALE = 20.0


class AnomalyLoop:
    def __init__(
        self,
        model,
        diffusion,
        data,
        batch_size,
        #lr,
        log_interval,
        lr_anneal_steps,
        save_interval,
        output_anomaly_name,
        data_obj,
        **kwargs,
    ):
        self.data_obj = data_obj
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        #self.lr = lr
        #self.lr_anneal_steps = lr_anneal_steps
        #self.opt = AdamW(self.model.parameters(), weight_decay=0.001)
        self.criterion = th.nn.MSELoss()
        self.step = 0
        self.loss = 0
        self.loss_acc = th.tensor([])
        self.steps_acc = th.tensor([])
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.weights = np.ones(diffusion.num_timesteps)
        #self.weights[0:25] /= 5
        #self.weights[25:50] /= 30
        #self.weights[50:75] /= 300
        #self.weights[75:100] /= 1000
        self.output_anomaly_name = output_anomaly_name
        self.load_model = kwargs['load_model']
        self.kwargs = kwargs
        self.loss_per_sample = th.zeros(len(self.data_obj),dtype=th.double)
        self.loss_per_sample2 = th.zeros(len(self.data_obj),dtype=th.double)
        self.occ_per_sample = th.zeros(len(self.data_obj))
        self.summary_file = self.init_summary_file()
        self.rpts = 2
        self.rpt_ind, self.rpt_weights, self.noise_array = self.sample_anomaly(rpts=self.rpts)
        print("hi")

    def auc(self, x, labels):
        if sum(x.values) != 0:
            fpr, tpr, thresholds = metrics.roc_curve(labels, -x, pos_label=0)
            auc = metrics.auc(fpr, tpr)
            return auc
        return 0

    def init_summary_file(self):
        columns = ["Index","Labels","Loss"]
        summary_file = pd.DataFrame(columns=columns)
        labels = self.data_obj.get_all_labels()
        summary_file = summary_file.assign(Index=np.arange(len(labels)), Labels=labels)
        return summary_file

    def run_loop(self):
        while (
            self.step <= self.data_obj.get_num_of_rows()
        ):
            batch,idx = next(self.data)
            self.run_step(batch,idx)
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


    def run_step(self, batch,idx):
        self.forward_backward(batch,idx,self.rpts)
        #self.opt.step()
        if self.step % self.log_interval == 0:
            self.log_step()
            self.loss_acc = th.cat((self.loss_acc,th.unsqueeze(self.loss, dim=0)),dim=0)
            self.steps_acc = th.cat((self.steps_acc,th.unsqueeze(th.tensor(self.step),dim=0)),dim=0)
            logger.dumpkvs()
        if self.step % self.data_obj.get_num_of_rows() == 0 and self.step != 0:

            self.create_summary_file()
            #self.save_model()

    def create_summary_file(self):
        mask = self.occ_per_sample != 0
        losses = th.zeros_like(self.loss_per_sample)
        #losses2 = th.zeros_like(self.loss_per_sample)
        losses[mask] = self.loss_per_sample[mask] / self.occ_per_sample[mask]
        #losses2[mask] = self.loss_per_sample2[mask] / self.occ_per_sample[mask]
        losses = self.normalize(losses)
        self.summary_file['Loss'] = losses
        #self.summary_file['Loss1'] = losses2
        t1 = self.summary_file['Loss']
        t2 = self.summary_file["Labels"]
        cur_auc = self.auc(t1,t2)
        print(cur_auc)
        summary_output_file_path = re.sub('.pt','_'+str(self.model.get_size())+'bit_'+str(self.step)+'steps_summary.csv',self.output_anomaly_name)
        self.summary_file.to_csv(summary_output_file_path,index=False)

    def normalize(self, input):
        if isinstance(input,np.ndarray):
            return (input-np.min(input))/(np.max(input)-np.min(input))
        return (input-th.min(input))/(max(input)-min(input))

    def save_model(self):
        kstep = int(self.step / 1000)
        if self.load_model != "":
            load_kstep = int(re.search(r"(\d+)kstep",self.load_model).group(1))
            kstep += load_kstep
        output_model_name = re.sub(r".pt", "_" + str(kstep) + "kstep.pt", self.output_model_name)
        th.save(self.model.state_dict(), output_model_name)

    def forward_backward(self, batch, idx, repeats_per_sample):
        #self.opt.zero_grad()
        #self.loss = self.criterion(batch, labels.float())

        for i in range(0, repeats_per_sample):
            #t, weights = self.sample(batch.shape[0])
            t,weights,noise = self.rpt_ind[i,None],self.rpt_weights[i,None],self.noise_array[i]
            noise = noise[:,None].T
            compute_losses = functools.partial(
                 self.diffusion.training_losses,
                 self.model,
                 batch,
                 t,
                 noise=noise
             )
            losses = compute_losses()
            self.occ_per_sample[idx] += 1
            self.loss_per_sample[idx] += losses['loss'].detach()
            self.loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                 self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

        self.loss_per_sample[idx] /= repeats_per_sample
            #self.loss.backward()

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
        #indices = 10*th.ones(8,dtype=int)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float()
        return indices, weights

    def sample_anomaly(self, rpts):
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
        #indices_np = np.random.choice(len(p), size=(rpts,), p=p)
        indices_np = np.linspace(0,2,rpts,dtype=int)
        indices = th.from_numpy(indices_np)
        #indices = 10*th.ones(8,dtype=int)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float()
        noise = []
        row_size = self.kwargs["row_size"]
        noise_t = th.randn_like(th.zeros(row_size))
        for i in range(rpts):
            #noise.append(th.randn_like(th.zeros(row_size)))
            noise.append(noise_t)
        return indices, weights, noise

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

