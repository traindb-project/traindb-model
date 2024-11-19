from copy import deepcopy
import os
import torch
import logging
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from stasy import sde_lib


def restore_checkpoint(ckpt_dir, state, device):
  if not os.path.exists(ckpt_dir):
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    try:
      state['epoch'] = loaded_state['epoch']
    except: pass
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step'],
    'epoch': state['epoch'],
  }
  torch.save(saved_state, ckpt_dir)


def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'sigmoid':
            ed = st + item[0]
            data_t.append(data[:,st:ed])
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.softmax(data[:, st:ed]))

            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)


def setup_sde(config):
    sde = None
    sampling_eps = 0

    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    return sde, sampling_eps


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    else:
        t = t.cpu()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list, loss_fn):

        self.model = model['model']
        self.optimizer = model['optimizer']
        self.dataset = dataset
        self.loss_fn = loss_fn

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        self.prune_per = 0.2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        self.optimizer.zero_grad()
        losses = self.loss_fn(self.model, self.dataset)
        loss = torch.mean(losses) # + penalty
        loss.backward()

        try:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)
        except:
            pass

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        self.model.train()

        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = torch.tensor(0., device=self.device, requires_grad=True)
        for n, p in model.named_parameters():
            if p.requires_grad:
                temp = torch.sum(self._precision_matrices[n] * (p - self._means[n]) ** 2)
                loss = loss + temp
        return loss

    def prune_weight(self, model: nn.Module):
        num_pruned_params = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                mask = self._precision_matrices[n] <= torch.quantile(self._precision_matrices[n], self.prune_per)
                p[mask] = 0. #  = torch.tensor(0., device='cuda')
                p = p.to_sparse()
                num_pruned_params += torch.sum(mask)
        return model, num_pruned_params

def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float):
    use_cuda = True if self.device == 'cuda' else False
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input, use_cuda), variable(target, use_cuda)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)
