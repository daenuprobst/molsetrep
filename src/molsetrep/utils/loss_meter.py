import torch
import numpy as np
from torcheval.metrics.metric import Metric


class LossMeter(Metric[torch.Tensor]):
    def __init__(self, device=None) -> None:
        super().__init__(device=device)
        self._add_state("losses", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, loss):
        self.losses = torch.cat((self.losses, torch.FloatTensor([loss])))
        return self

    @torch.inference_mode()
    def compute(self):
        # Let scipy do the hard work
        return np.sum(self.losses.cpu().detach().numpy()) / self.losses.shape[0]

    @torch.inference_mode()
    def merge_state(self, metrics):
        losses = [self.losses]

        for metric in metrics:
            losses.append(metric.losses)
        self.losses = torch.cat(losses)
        return self
