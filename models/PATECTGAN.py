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
# import rdt
# import sdv
from TrainDBBaseModel import TrainDBSynopsisModel
import pandas as pd
import torch

#import numpy as np
import os
from snsynth.pytorch.nn import PATECTGAN as ppateCTGAN
from snsynth.pytorch.pytorch_synthesizer import PytorchDPSynthesizer
# from snsynth.pytorch.nn.ctgan.base import BaseSynthesizer

LOGGER = logging.getLogger(__name__)

class PATECTGAN(TrainDBSynopsisModel, PytorchDPSynthesizer):

    def __init__(self, **kwargs):
        self.model_kwargs = kwargs

    def train(self, real_data, table_metadata):
        self.columns, categorical_columns = self.get_columns(real_data, table_metadata)
        #real_data = real_data[columns]

        LOGGER.info("Training %s", self.__class__.__name__)
        self.model = PytorchDPSynthesizer(1.0, ppateCTGAN(regularization='dragan'), None)
        # self.model = sdv.tabular.CTGAN(table_metadata=table_metadata, **self.model_kwargs)
        self.model.fit(real_data, categorical_columns=real_data.columns.values.tolist()) 

    def save(self, output_path):
        #self.model = BaseSynthesizer.save(output_path + '/model.pkl') # pateCTGAN에는 save 함수가 없음
        torch.save({
            'model': self.model,
            'columns': self.columns
        }, os.path.join(output_path, 'model_info.pth'))

    def load(self, input_path):
        #self.model = BaseSynthesizer.load(input_path + '/model.pkl')
        saved_model = torch.load(os.path.join(input_path, 'model_info.pth'))
        self.model = saved_model['model']
        self.columns = saved_model['columns']

    def synopsis(self, row_count):
        LOGGER.info("Synopsis Generating %s", self.__class__.__name__)
        #gan = PytorchDPSynthesizer.__init__
        #self.model = PytorchDPSynthesizer(1.0, ppateCTGAN(regularization='dragan'), None)
        synthetic_data = self.model.sample(row_count)
        synthetic_data = pd.DataFrame(synthetic_data, columns=self.columns)

        return synthetic_data
