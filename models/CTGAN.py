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

import ast
import logging
import os
import rdt
import sdv
import pandas as pd
import torch
from TrainDBBaseModel import TrainDBModel, TrainDBSynopsisModel

LOGGER = logging.getLogger(__name__)

class CTGAN(TrainDBSynopsisModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("verbose", "True")
        for key, value in kwargs.items():
          try:
            kwargs[key] = ast.literal_eval(value)
          except (ValueError, SyntaxError):
            pass

        self.model_kwargs = kwargs

    def train(self, real_data, table_metadata):
        self.columns, _ = self.get_columns(real_data, table_metadata)

        LOGGER.info("Training %s", self.__class__.__name__)
        self.model = sdv.tabular.CTGAN(table_metadata=table_metadata, **self.model_kwargs)
        self.model.fit(real_data)

    def save(self, output_path):
        self.model.save(os.path.join(output_path, 'model.pkl'))
        torch.save({
            'columns': self.columns
        }, os.path.join(output_path, 'model_info.pth'))

    def load(self, input_path):
        self.model = sdv.tabular.CTGAN.load(os.path.join(input_path, 'model.pkl'))
        saved_model_info = torch.load(os.path.join(input_path, 'model_info.pth'))
        self.columns = saved_model_info['columns']

    def list_hyperparameters():
        hparams = []
        hparams.append(TrainDBModel.createHyperparameter('embedding_dim', 'int', '128', 'the size of the random sample passed to the generator'))
        hparams.append(TrainDBModel.createHyperparameter('generator_dim', 'tuple or list of ints', '(256, 256)', 'the size of the output samples for each one of the residuals'))
        hparams.append(TrainDBModel.createHyperparameter('discriminator_dim', 'tuple or list of ints', '(256, 256)', 'the size of the output samples for each one of the discriminator layers'))
        hparams.append(TrainDBModel.createHyperparameter('generator_lr', 'float', '2e-4', 'the learning rate for the generator'))
        hparams.append(TrainDBModel.createHyperparameter('generator_decay', 'float', '1e-6', 'generator weight decay for the Adam optimizer'))
        hparams.append(TrainDBModel.createHyperparameter('discriminator_lr', 'float', '2e-4', 'the learning rate for the discriminator'))
        hparams.append(TrainDBModel.createHyperparameter('discriminator_decay', 'float', '1e-6', 'discriminator weight decay for the Adam optimizer'))
        hparams.append(TrainDBModel.createHyperparameter('batch_size', 'int', '500', 'the number of samples to process in each step'))
        hparams.append(TrainDBModel.createHyperparameter('epochs', 'int', '300', 'the number of training epochs'))
        hparams.append(TrainDBModel.createHyperparameter('pac', 'int', '10', 'the number of samples to group together when applying the discriminator'))
        return hparams

    def synopsis(self, row_count):
        LOGGER.info("Synopsis Generating %s", self.__class__.__name__)
        synthetic_data = self.model.sample(row_count)
        synthetic_data = pd.DataFrame(synthetic_data, columns=self.columns)

        return synthetic_data
