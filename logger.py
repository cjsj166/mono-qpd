import os
import logging
import torch
import numpy
from torch.utils.tensorboard import SummaryWriter


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler, total_steps, log_dir='result/runs'):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = total_steps
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir))

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join('result/runs'))

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join('result/runs'))

        for key in results:
            
            if isinstance(results[key], torch.Tensor):
                if results[key].dim() == 4:
                    results[key] = results[key][0]
                self.writer.add_image(key, results[key], self.total_steps)
            elif isinstance(results[key], numpy.ndarray):
                if results[key].ndim == 4:
                    results[key] = results[key][0]
                self.writer.add_image(key, results[key], self.total_steps)
            else:
                self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
