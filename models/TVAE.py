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

import logging
import rdt
import sdv
from TrainDBBaseModel import TrainDBSynopsisModel
import pandas as pd

import torch

LOGGER = logging.getLogger(__name__)

class TVAE(TrainDBSynopsisModel):

    def __init__(self, **kwargs):
        self.model_kwargs = kwargs

    def train(self, real_data, table_metadata):
        self.columns, _ = self.get_columns(real_data, table_metadata)

        LOGGER.info("Training %s", self.__class__.__name__)
        self.model = sdv.tabular.TVAE(table_metadata=table_metadata, **self.model_kwargs)
        self.model.fit(real_data)

    def save(self, output_path):
        self.model.save(output_path + '/model.pkl')
        torch.save({
            'columns': self.columns
        }, output_path + '/model_info.pth')

    def load(self, input_path):
        self.model = sdv.tabular.TVAE.load(input_path + '/model.pkl')
        saved_model_info = torch.load(input_path + '/model_info.pth')
        self.columns = saved_model_info['columns']

    def synopsis(self, row_count):
        LOGGER.info("Synopsis Generating %s", self.__class__.__name__)
        synthetic_data = self.model.sample(row_count)
        synthetic_data = pd.DataFrame(synthetic_data, columns=self.columns)

        return synthetic_data
