from torch.optim.lr_scheduler import LRScheduler


class WarmupExponentialLR(LRScheduler):

    def __init__(self, optimizer, gamma, warmup_step, verbose="deprecated"):
        self.gamma = gamma
        self.warmup_step = warmup_step
        super().__init__(optimizer, -1, verbose)

    def get_lr(self):

        if self._step_count < self.warmup_step:
            ratio = float(self._step_count) / float(self.warmup_step)
            return [group['initial_lr'] * ratio
                    for group in self.optimizer.param_groups]


        else:
            return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

