from torch.optim import Optimizer

from .drsom import DRSOM
from .LBFGS_ps import LBFGS

class DRSOM_LBFGS(Optimizer):
    def __init__(self, params, switch_epochs, adam_params, lbfgs_params):
        # defaults = dict(switch_epoch=switch_epoch, adam_params=adam_params, lbfgs_params=lbfgs_params)

        self.switch_epochs = sorted(switch_epochs)
        self.params = list(params)
        self.drsom = DRSOM(self.params, lr=1e-4)
        self.lbfgs_params = lbfgs_params
        # self.lbfgs = LBFGS(self.params, **lbfgs_params)

        super(DRSOM_LBFGS, self).__init__(self.params, defaults={})

        self.state['epoch'] = 0

    def step(self, closure=None):
        if self.state['epoch'] < self.switch_epochs[0]:
            self.drsom.step(closure)
        else:
            # (Re)start LBFGS optimizer
            if self.state['epoch'] in self.switch_epochs:
                print(f'Starting LBFGS optimizer at epoch {self.state["epoch"]}')
                self.lbfgs = LBFGS(self.params, max_iter=1, **self.lbfgs_params)
            self.lbfgs.step(closure)

        self.state['epoch'] += 1