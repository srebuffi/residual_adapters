# sgd.py
# created by Sylvestre-Alvise Rebuffi [srebuffi@robots.ox.ac.uk]
# Copyright © The University of Oxford, 2020
# This code is made available under the Apache v2.0 licence, see LICENSE.txt for details

import torch
import math
import torch.nn.functional as F
import config_task

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                siz = p.grad.size()
                if len(siz) > 3:
                    if siz[2] == 3:
                        weight_decay = config_task.decay3x3[config_task.task]
                    elif siz[2] == 1:
                        weight_decay = config_task.decay1x1[config_task.task]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
                p.data.add_(-group['lr'], d_p)

        return loss

