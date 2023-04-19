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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import numpy as np
import tensorflow as tf
import pandas as pd
import logging
from models import ncsnpp_tabular
import losses
import likelihood
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
from torch.utils.data import DataLoader
import evaluation
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from utils import save_checkpoint, restore_checkpoint, apply_activate
import collections
import os

FLAGS = flags.FLAGS


def train(config, workdir):
  randomSeed = 2021
  torch.manual_seed(randomSeed)
  torch.cuda.manual_seed(randomSeed)
  torch.cuda.manual_seed_all(randomSeed) 
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(randomSeed)
  tf.random.set_seed(randomSeed)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  num_params = sum(p.numel() for p in score_model.parameters())
  print("the number of parameters", num_params)

  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))

  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['epoch'])

  # Build data iterators
  train_ds, eval_ds, (transformer, meta) = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization) 

  if meta['problem_type'] == 'binary_classification': 
    metric = 'binary_f1'
  elif meta['problem_type'] == 'regression': metric = "r2"
  else: metric = 'macro_f1'

  logging.info(f"train shape : {train_ds.shape}")
  logging.info(f"eval.shape : {eval_ds.shape}")  

  logging.info(f"batch size: {config.training.batch_size}")
  train_ds_ = transformer.inverse_transform(train_ds)
  if metric != "r2":
    logging.info('raw data : {}'.format(collections.Counter(train_ds_[:,-1])))                                          

  train_iter = DataLoader(train_ds, batch_size=config.training.batch_size)
  eval_iter = iter(DataLoader(eval_ds, batch_size=config.eval.batch_size))  # pytype: disable=wrong-arg-types

  scaler = datasets.get_data_scaler(config) 
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
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
  logging.info(score_model)

  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting

  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting, workdir=workdir, spl=config.training.spl, writer=writer, 
                                     alpha0=config.model.alpha0, beta0=config.model.beta0)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting, workdir=workdir, spl=config.training.spl, writer=writer, 
                                     alpha0=config.model.alpha0, beta0=config.model.beta0)
  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  test_iter = config.test.n_iter

  logging.info("Starting training loop at epoch %d." % (initial_step,))
  scores_max = 0

  likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)
  sampling_shape = (train_ds.shape[0], config.data.image_size)
  
  for epoch in range(initial_step, config.training.epoch+1):
    state['epoch'] += 1
    for iteration, batch in enumerate(train_iter): 
      batch = batch.to(config.device).float()
      loss = train_step_fn(state, batch) 
      writer.add_scalar("training_loss", loss.item(), state['step'])

    logging.info("epoch: %d, iter: %d, training_loss: %.5e" % (epoch, iteration, loss.item()))

    if epoch == 0 or epoch % 1 == 0: 
      ema.store(score_model.parameters())
      ema.copy_to(score_model.parameters())
      sample, n = sampling_fn(score_model, sampling_shape=sampling_shape)
      sample = apply_activate(sample, transformer.output_info)
      ema.restore(score_model.parameters())

      train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                      uniform_dequantization=True, evaluation=True)
      sample = transformer.inverse_transform(sample.cpu().numpy())
      eval_samples = transformer.inverse_transform(eval_ds_bpd)

      scores, _ = evaluation.compute_scores([eval_samples], [sample], meta)
      if metric != "r2":
        logging.info('sampling data : {}'.format(collections.Counter(sample[:,-1])))

      f1 = scores.mean(axis=0)[metric]
      logging.info(f"epoch: {epoch}, {metric}: {f1}")
      writer.add_scalar(metric, torch.tensor(f1), epoch) 

      if scores_max < torch.tensor(f1):
        scores_max = torch.tensor(f1)
        save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_max.pth'), state)

      save_checkpoint(checkpoint_meta_dir, state) # save the model each epoch

  test_iter = 5
  train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config, uniform_dequantization=True, evaluation=True)
  ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_max.pth")
  state = restore_checkpoint(ckpt_filename, state, device=config.device)
  logging.info(f"checkpoint : {state['step']}")
  ema.copy_to(score_model.parameters())

  sample_list = []
  num_sampling_rounds = 5

  for r in range(num_sampling_rounds):   
    samples, n = sampling_fn(score_model, sampling_shape=sampling_shape)
    samples = apply_activate(samples, transformer.output_info)
  
    samples = transformer.inverse_transform(samples.cpu().numpy())
    sample_list.append(samples)
    pd.DataFrame(samples).to_csv(f"{workdir}/samples/{r}.csv")

  eval_samples = transformer.inverse_transform(eval_ds_bpd)
    
  scores, _ = evaluation.compute_scores([eval_samples]*num_sampling_rounds, sample_list, meta)
  pd.DataFrame(scores).to_csv(f"{workdir}/results.csv")
  # score_list.append(scores[metric].mean())

  logging.info(f"{scores}")


  


def eval(config, workdir):
  randomSeed = 2022
  torch.manual_seed(randomSeed)
  torch.cuda.manual_seed(randomSeed)
  torch.cuda.manual_seed_all(randomSeed) 
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(randomSeed)
  tf.random.set_seed(randomSeed)

  # Initialize model.
  score_model = mutils.create_model(config)
  print(score_model)
  num_params = sum(p.numel() for p in score_model.parameters())
  print("the number of parameters", num_params)

  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints", "checkpoint_finetune.pth")
  samples_dir = os.path.join(workdir, "samples")

  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  tf.io.gfile.makedirs(samples_dir)

  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['epoch'])

  train_ds, eval_ds, (transformer, meta) = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization) 

  if meta['problem_type'] == 'binary_classification': 
    metric = 'binary_f1'
  elif meta['problem_type'] == 'regression':
    metric = 'r2'
  else: metric = 'macro_f1'


  test_iter = 5
  train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                        uniform_dequantization=True, evaluation=True)
  logging.info(f"checkpoint : {state['step']}")
  ema.copy_to(score_model.parameters())

  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
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

  logging.info(score_model)
  sample_list = []
  num_sampling_rounds = 5

  sampling_shape = (train_ds.shape[0], config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)


  for r in range(num_sampling_rounds):   
    samples, n = sampling_fn(score_model, sampling_shape=sampling_shape)
    samples = apply_activate(samples, transformer.output_info)
  
    samples = transformer.inverse_transform(samples.cpu().numpy())
    sample_list.append(samples)
    pd.DataFrame(samples).to_csv(f"{workdir}/samples/{r}.csv")

  eval_samples = transformer.inverse_transform(eval_ds_bpd)
    
  scores, _ = evaluation.compute_scores([eval_samples]*num_sampling_rounds, sample_list, meta)
  pd.DataFrame(scores).to_csv(f"{workdir}/results.csv")

  logging.info(f"{scores}")




def fine_tune(config, workdir):
  randomSeed = 2022
  torch.manual_seed(randomSeed)
  torch.cuda.manual_seed(randomSeed)
  torch.cuda.manual_seed_all(randomSeed) 
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(randomSeed)
  tf.random.set_seed(randomSeed)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  num_params = sum(p.numel() for p in score_model.parameters())
  print("the number of parameters", num_params)

  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  samples_dir = os.path.join(workdir, "samples")

  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  tf.io.gfile.makedirs(samples_dir)

  train_ds, eval_ds, (transformer, meta) = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization)

  if meta['problem_type'] == 'binary_classification': metric = 'binary_f1'
  elif meta['problem_type'] == 'multiclass_classification': metric = 'macro_f1'
  else: metric = 'r2'

  logging.info(f"train shape : {train_ds.shape}")
  logging.info(f"eval.shape : {eval_ds.shape}")  

  train_ds_ = transformer.inverse_transform(train_ds)
  if metric !='r2':
    logging.info('raw data : {}'.format(collections.Counter(train_ds_[:,-1])))                                          

  train_iter = iter(DataLoader(train_ds, batch_size=config.training.batch_size))

  scaler = datasets.get_data_scaler(config) 
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
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
  logging.info(score_model)

  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting, workdir=workdir, spl=False, writer=writer, 
                                     alpha0=config.model.alpha0, beta0=config.model.beta0)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting, workdir=workdir, spl=False, writer=writer, 
                                     alpha0=config.model.alpha0, beta0=config.model.beta0)

  if config.training.snapshot_sampling:
    sampling_shape = (train_ds.shape[0], config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  test_iter = config.test.n_iter

  ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_max.pth")
  state = restore_checkpoint(ckpt_filename, state, device=config.device)
  logging.info(f"checkpoint : {state['step']}")
  ema.copy_to(score_model.parameters())

  num_sampling_rounds = 5

  hutchinson_type = config.training.hutchinson_type
  tolerance = config.training.tolerance
  
  likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler, hutchinson_type, tolerance, tolerance)

  train_ds = torch.tensor(train_ds, device=config.device, dtype=torch.float32)
  train_ll = likelihood_fn(score_model, train_ds, eps_iters = config.training.eps_iters)[0]

  if config.training.retrain_type == 'median':
    idx = torch.where(train_ll <= torch.median(train_ll), True, False)
  elif config.training.retrain_type == 'mean':
    idx = torch.where(train_ll <= torch.mean(train_ll), True, False)

  logging.info(f"log likelihood mean: {torch.mean(train_ll)}, median : {torch.median(train_ll)}, std : {torch.std(train_ll)}")

  re_train = train_ds[idx]
  
  logging.info(f"the number of re-train: {len(re_train)} / {len(train_ll)}")

  train_iter = DataLoader(re_train, batch_size=config.training.batch_size)
  step = 0

  samples, n = sampling_fn(score_model, sampling_shape=sampling_shape)
  samples = apply_activate(samples, transformer.output_info)
  samples = transformer.inverse_transform(samples.cpu().numpy())
  scores_max = 0

  for epoch in range(config.training.fine_tune_epochs):
    logging.info("----------- epoch %d START ----------" % (epoch) )

    for iteration, batch in enumerate(train_iter):
      batch = batch.to(config.device).float()
      
      loss = train_step_fn(state, batch)
      logging.info("epoch: %d, iter: %d, training_loss: %.5e" % (epoch, iteration, loss.item()))
      writer.add_scalar("training_loss", loss, step)
      step += step 
    
    logging.info("----------- epoch %d END ----------" % (epoch) )
    
    train_ll_after = likelihood_fn(score_model, train_ds, eps_iters = config.training.eps_iters)[0]

    logging.info(f"epoch {epoch} log likelihood mean: {torch.mean(train_ll_after)}, median : {torch.median(train_ll_after)}, std : {torch.std(train_ll_after)}")

    diff = train_ll_after - train_ll
    idx_after = torch.where(diff < -0.1, True, False)
    re_train = train_ds[idx_after]
    logging.info(f"the number of decreased likelihood: {len(re_train)} / {len(train_ll)}")

    train_iter = DataLoader(re_train, batch_size=config.training.batch_size)

    logging.info(f"epoch : {epoch} --- sampling")
    
    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                          uniform_dequantization=True, evaluation=True)

    samples, n = sampling_fn(score_model, sampling_shape=sampling_shape)
    samples = apply_activate(samples, transformer.output_info)
    samples = transformer.inverse_transform(samples.cpu().numpy())

    sample_list = []

    for r in range(num_sampling_rounds):   

      samples, n = sampling_fn(score_model, sampling_shape=sampling_shape)
      samples = apply_activate(samples, transformer.output_info)
    
      samples = transformer.inverse_transform(samples.cpu().numpy())
      sample_list.append(samples)
      # pd.DataFrame(samples).to_csv(f"{workdir}/samples/after_fune_tune_{r}.csv")

    eval_samples = transformer.inverse_transform(eval_ds_bpd)
      
    if metric !='r2':
      logging.info('sampling data : {}'.format(collections.Counter(samples[:,-1]))) 

    scores, _ = evaluation.compute_scores([eval_samples]*num_sampling_rounds, sample_list, meta)
    # pd.DataFrame(scores).to_csv(f"{workdir}/results.csv")
  
    logging.info(f"{scores}")

    if scores_max < torch.tensor(scores.mean(axis=0)[metric]):
        scores_max = torch.tensor(scores.mean(axis=0)[metric])
        save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_finetune.pth'), state)

