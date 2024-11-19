# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

from pickle import FALSE
import torch
import torch.optim as optim
import numpy as np
from stasy.models import utils as mutils
from stasy.sde_lib import VESDE, VPSDE
import logging

def get_optimizer(config, params):
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum((step+1) / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()
  return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None] * z

    score = score_fn(perturbed_data, t)
    
    if not likelihood_weighting:
      losses = torch.square(score * std[:, None] + z) 
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    # loss = torch.mean(losses)
    return losses

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, likelihood_weighting=False):

  loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                            continuous=True, likelihood_weighting=likelihood_weighting)

  def step_fn(state, batch, ewc=None, importance=100):
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      losses = loss_fn(model, batch)
      loss = torch.mean(losses) # + penalty
      loss.backward()

      if ewc:
        penalty = importance * ewc.penalty(model)
        penalty.backward()

      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())


    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        losses, score = loss_fn(model, batch)
        ema.restore(model.parameters())
        loss = torch.mean(losses)

    return loss

  return step_fn
