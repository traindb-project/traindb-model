"""
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


from absl import flags
import collections
import logging
import numpy as np
import os
import pandas as pd
import torch 
from torch.utils.data import DataLoader

from stasy import datasets, losses, sampling
from stasy.configs.default_tabular_configs import get_default_configs
from stasy.utils import apply_activate, setup_sde
from stasy.models import utils as mutils
from stasy.models.ema import ExponentialMovingAverage
from stasy.models.tabular_utils import GeneralTransformer

from TrainDBBaseModel import TrainDBModel, TrainDBSynopsisModel

LOGGER = logging.getLogger(__name__)

class STASY(TrainDBSynopsisModel):

    def __init__(self,
                 batch_size=1000,
                 epochs=10000):
        self.config = get_default_configs()
        self.config.batch_size = batch_size
        self.config.training.epoch = epochs
        self.transformer = GeneralTransformer()
        self.columns = []
        self.state = {}

    def train(self, real_data, table_metadata):
        columns, categoricals = self.get_columns(real_data, table_metadata)
        real_data = real_data[columns]
        self.transformer.meta = GeneralTransformer.get_metadata(real_data)
        self.columns = columns
        
        self.config.data.image_size = len(real_data.columns)
        config = self.config

        score_model = mutils.create_model(config)
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        self.state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)
        state = self.state;

        initial_step = int(state['epoch'])
        train_ds = self.transformer.transform(real_data.to_numpy())
        metric = 'macro_f1'

        logging.info(f"train shape : {train_ds.shape}")
        logging.info(f"batch size: {config.training.batch_size}")

        train_iter = DataLoader(train_ds, batch_size=config.training.batch_size)
        scaler = datasets.get_data_scaler(config) 
        inverse_scaler = datasets.get_data_inverse_scaler(config)

        logging.info(score_model)

        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        reduce_mean = config.training.reduce_mean
        likelihood_weighting = config.training.likelihood_weighting

        sde, sampling_eps = setup_sde(config)

        train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                           reduce_mean=reduce_mean, continuous=continuous,
                                           likelihood_weighting=likelihood_weighting,
                                           spl=config.training.spl,
                                           alpha0=config.model.alpha0, beta0=config.model.beta0)

        logging.info("Starting training loop at epoch %d." % (initial_step,))
        scores_max = 0

        for epoch in range(initial_step, config.training.epoch):
            state['epoch'] += 1
            for iteration, batch in enumerate(train_iter): 
                batch = batch.to(config.device).float()
                loss = train_step_fn(state, batch) 
       
            logging.info("epoch: %d, iter: %d, training_loss: %.5e" % (epoch, iteration, loss.item()))
            print("epoch: %d, iter: %d, training_loss: %.5e" % (epoch, iteration, loss.item()))
       
    def save(self, output_path):
        state = self.state
        saved_state = {
          'optimizer': state['optimizer'].state_dict(),
          'model': state['model'].state_dict(),
          'ema': state['ema'].state_dict(),
          'step': state['step'],
          'epoch': state['epoch'],
          'config': self.config,
          'transformer': self.transformer,
          'columns': self.columns,
        }
        torch.save(saved_state, os.path.join(output_path, 'model.pth'))

    def load(self, input_path):
        self.state = torch.load(os.path.join(input_path, 'model.pth'))
        self.config = self.state['config']
        self.transformer = self.state['transformer']
        self.columns = self.state['columns']

    def synopsis(self, row_count):

        config = self.config
        transformer = self.transformer
        score_model = mutils.create_model(config)

        sde, sampling_eps = setup_sde(config)
        inverse_scaler = datasets.get_data_inverse_scaler(config)
        sampling_shape = (row_count, len(self.columns))
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

        samples, n = sampling_fn(score_model, sampling_shape=sampling_shape)
        samples = apply_activate(samples, transformer.output_info)
        samples = transformer.inverse_transform(samples.cpu().numpy())

        synthetic_data = pd.DataFrame(samples, columns=self.columns)

        return synthetic_data

