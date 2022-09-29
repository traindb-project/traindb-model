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

from TrainDBBaseModel import TrainDBModel
from rspn.ensemble_compilation.spn_ensemble import SPNEnsemble, read_ensemble
from rspn.ensemble_compilation.graph_representation import SchemaGraph, Table
from rspn.evaluation.utils import parse_query
from rspn.aqp_spn.aqp_spn import AQPSPN
from spn.structure.StatisticalTypes import MetaType
import numpy as np
import torch

class RSPN(TrainDBModel):

    def __init__(self,
                 strategy='single',
                 rdc_threshold=0.3,
                 samples_per_spn=10000000,
                 bloom_filters=True,
                 max_rows_per_hdf_file=20000000,
                 post_sampling_factor=30,
                 incremental_learning_rate=0,
                 incremental_condition='',
                 epochs=0):
        self.strategy = strategy
        self.rdc_threshold = rdc_threshold
        self.samples_per_spn = samples_per_spn
        self.bloom_filters = bloom_filters
        self.max_rows_per_hdf_file = max_rows_per_hdf_file
        self.post_sampling_factor = post_sampling_factor
        self.incremental_learning_rate = incremental_learning_rate
        self.incremental_condition = incremental_condition
        self.columns = []

    def train(self, real_data, table_metadata):
        columns, categoricals = self.get_columns(real_data, table_metadata)
        real_data = real_data[columns]
        self.columns = columns
        table_size = len(real_data)

        schema = SchemaGraph()
        schema.add_table(Table(table_metadata['table'], attributes=columns, table_size=table_size))
        spn_ensemble = SPNEnsemble(schema)

        meta_types = []
        null_values = []
        for col in columns:
            if col in categoricals:
                meta_types.append(MetaType.DISCRETE)
            else:
                meta_types.append(MetaType.REAL)
            null_values.append(None)
        full_join_size = table_size
        full_sample_size = full_join_size

        table_set = set()
        for table in schema.tables:
            table_set.add(table.table_name)

        table = schema.table_dictionary[table_metadata['table']].table_name
        rspn_columns = [table + '.' + col for col in columns]
        real_data.columns = rspn_columns

        aqp_spn = AQPSPN(meta_types, null_values, full_join_size, schema, None, full_sample_size,
                         table_set=table_set, column_names=rspn_columns)
        aqp_spn.learn(real_data.to_numpy(), rdc_threshold=self.rdc_threshold)

        spn_ensemble.add_spn(aqp_spn)

        self.schema = schema
        self.spn_ensemble = spn_ensemble

    def save(self, output_path):
        torch.save({
            'schema': self.schema
        }, output_path + 'model.pth')
        self.spn_ensemble.save(output_path + "spn_ensembles")

    def load(self, input_path):
        saved_model = torch.load(input_path + 'model.pth')
        self.schema = saved_model['schema']
        self.spn_ensemble = read_ensemble(input_path + "spn_ensembles")

    def infer(self, query_string):
        query = parse_query(query_string.strip(), self.schema)
        confidence_intervals, aqp_result = self.spn_ensemble.evaluate_query(query,
                                        confidence_sample_size=10000,
                                        confidence_intervals=True)
        print("confidence intervals:", confidence_intervals)
        print("aqp_result:", aqp_result)
        return aqp_result
