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
"""Return training and evaluation/test datasets from config files."""
import torch
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from models.tabular_utils import GeneralTransformer
from datasets_tabular import load_data


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def get_dataset(config, uniform_dequantization=False, evaluation=False):
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size

  if batch_size % torch.cuda.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({torch.cuda.device_count()})')


  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1

  # Create dataset builders for tabular data.
  train, test, cols = load_data(config.data.dataset)
  transformer = GeneralTransformer()
  data = np.concatenate([train, test])
  transformer.fit(data, cols[0], cols[1])

  train_data = transformer.transform(train)
  test = transformer.transform(test)

  return train_data, test, (transformer, cols[2])
    