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

from TrainDBBaseModel import TrainDBModel, TrainDBInferenceModel
from rspn.ensemble_compilation.spn_ensemble import SPNEnsemble, read_ensemble
from rspn.ensemble_compilation.graph_representation import SchemaGraph, Table
from rspn.aqp_spn.aqp_spn import AQPSPN
from rspn.evaluation.utils import handle_aggregation
from rspn.ensemble_compilation.graph_representation import Query, QueryType
from spn.structure.StatisticalTypes import MetaType
import numpy as np
import os
import sqlparse
import torch

class RSPN(TrainDBInferenceModel):

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
        self.schema = None
        self.spn_ensemble = None

    def train(self, real_data, table_metadata):
        """
        train a model from real_data(.csv) and table_metadata(.json)
        :return : a learned SPNEnsemble object (which can be saved as .pth file using the save() method)
        """
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

        rspn_table_metadata = dict()
        table_set = set()
        for table in schema.tables:
            table_set.add(table.table_name)
            rspn_table_metadata[table.table_name] = dict()

        table_obj = schema.table_dictionary[table_metadata['table']]
        table = table_obj.table_name
        rspn_columns = [table + '.' + col for col in columns]
        real_data.columns = rspn_columns

        relevant_attributes = [x for x in table_obj.attributes if x not in table_obj.irrelevant_attributes]
        rspn_table_metadata, real_data, relevant_attributes = self.manage_functional_dependencies(
            table, table_obj, real_data, rspn_table_metadata, relevant_attributes)

        # save if there are entities without FK reference (e.g. orders without customers)
        outgoing_relationships = self.find_relationships(schema, table, incoming=False)
        for relationship_obj in outgoing_relationships:
            fk_attribute_name = table + '.' + relationship_obj.start_attr

            rspn_table_metadata[relationship_obj.identifier] = {
                'fk_attribute_name': fk_attribute_name,
                'length': real_data[fk_attribute_name].isna().sum(),
                # 'length': table_data[fk_attribute_name].isna().sum() * 1 / table_sample_rate,
                'path': None
            }

        real_data, rspn_table_metadata, relevant_attributes, del_cat_attributes = \
            self.impute_null_value_and_replace_categorical_value(
                table, real_data, rspn_table_metadata, relevant_attributes, 10000)

        aqp_spn = AQPSPN(meta_types, null_values, full_join_size, schema, None, full_sample_size,
                         table_set=table_set, column_names=rspn_columns, table_meta_data=rspn_table_metadata)
        aqp_spn.learn(real_data.to_numpy(), rdc_threshold=self.rdc_threshold)

        spn_ensemble.add_spn(aqp_spn)

    def save(self, output_path):
        torch.save({
            'schema': self.schema
        }, os.path.join(output_path, 'model.pth'))
        self.spn_ensemble.save(os.path.join(output_path, 'spn_ensembles'))

    def load(self, input_path):
        saved_model = torch.load(os.path.join(input_path, 'model.pth'))
        self.schema = saved_model['schema']
        self.spn_ensemble = read_ensemble(os.path.join(input_path, 'spn_ensembles'), True)

    def infer(self, agg_expr, group_by_column, where_condition):
        query = Query(self.schema)

        alias_dict = dict()
        for table in self.schema.tables:
            query.table_set.add(table.table_name)
            alias_dict[table.table_name] = table.table_name
        table = self.schema.tables[0].table_name

        if group_by_column:
            query.add_group_by(table, group_by_column)

        if where_condition:
            query.add_where_condition(table, where_condition)

        total_aqp_results = []
        total_confidence_itvs = []
        agg_list = agg_expr.split(",")
        for agg in agg_list:
            if agg.lower() == 'count(*)':
                query.query_type = QueryType.CARDINALITY
            else:
                query.query_type = QueryType.AQP
                handle_aggregation(alias_dict, query, self.schema, sqlparse.parse(agg)[0])

            agg_confidence_itvs, agg_aqp_results = self.spn_ensemble.evaluate_query(query,
                                            confidence_sample_size=10000,
                                            confidence_intervals=True)
            if len(total_aqp_results) > 0:
                if group_by_column:
                    agg_aqp_results = np.delete(agg_aqp_results, 0, 1)
                total_aqp_results = np.append(total_aqp_results, np.atleast_2d(agg_aqp_results), axis=1)
                total_confidence_itvs = list(zip(total_confidence_itvs, np.atleast_2d(agg_confidence_itvs)))
            else:
                total_aqp_results = np.atleast_2d(agg_aqp_results)
                total_confidence_itvs = np.atleast_2d(agg_confidence_itvs)

        return np.atleast_2d(total_aqp_results), np.atleast_2d(total_confidence_itvs)

    def list_hyperparameters():
        hparams = []
        hparams.append(TrainDBModel.createHyperparameter('strategy', 'str', 'single', 'type of RSPN ensemble'))
        hparams.append(TrainDBModel.createHyperparameter('rdc_threshold', 'float', '0.3', 'threshold for determining correlation'))
        hparams.append(TrainDBModel.createHyperparameter('sample_per_spn', 'int', '10000000', 'the number of samples per spn'))
        hparams.append(TrainDBModel.createHyperparameter('bloom_filters', 'bool', 'True', 'whether bloom_filter applies or not'))
        hparams.append(TrainDBModel.createHyperparameter('max_rows_per_hdf_file', 'int', '20000000', 'max rows per hdf file'))
        hparams.append(TrainDBModel.createHyperparameter('post_sampling_factor', 'int', '30', 'post sampling factor'))
        hparams.append(TrainDBModel.createHyperparameter('incremental_learning_rate', 'int', '0', 'init learning / incremental'))
        hparams.append(TrainDBModel.createHyperparameter('incremental_condition', 'str', '', 'predicate for incremental learning'))
        hparams.append(TrainDBModel.createHyperparameter('epochs', 'int', '0', 'the number of training epochs'))
        return hparams

    def manage_functional_dependencies(self, table, table_obj, table_data, table_meta_data, relevant_attributes):
        """
        Manage functional dependencies
        * Refactored(Extract Function) from prepare_single_table [kihyuk-nam:2022.08.17]
        :return: table_meta_data, table_data, relevant_attributes
        """
        # logger.info(f"Managing functional dependencies for table {table}")
        table_meta_data[table] = dict()
        table_meta_data[table]['fd_dict'] = dict()
        cols_to_be_dropped = []
        for attribute_wo_table in table_obj.attributes:
            attribute = table + '.' + attribute_wo_table
            fd_children = table_obj.children_fd_attributes(attribute)
            if len(fd_children) > 0:
                for child in fd_children:
                    # logger.info(f"Managing functional dependencies for {child}->{attribute}")
                    distinct_tuples = table_data.drop_duplicates([attribute, child])[[attribute, child]].values
                    reverse_dict = {}
                    for attribute_value, child_value in distinct_tuples:
                        if reverse_dict.get(attribute_value) is None:
                            reverse_dict[attribute_value] = []
                        reverse_dict[attribute_value].append(child_value)
                    if table_meta_data[table]['fd_dict'].get(attribute) is None:
                        table_meta_data[table]['fd_dict'][attribute] = dict()
                    table_meta_data[table]['fd_dict'][attribute][child] = reverse_dict
                # remove from dataframe and relevant attributes
                cols_to_be_dropped.append(attribute)
                relevant_attributes.remove(attribute_wo_table)
        table_data.drop(columns=cols_to_be_dropped, inplace=True)

        return table_meta_data, table_data, relevant_attributes

    def find_relationships(self, schema_graph, table, incoming=True):
        relationships = []

        for relationship_obj in schema_graph.relationships:

            if relationship_obj.end == table and incoming:
                relationships.append(relationship_obj)
            if relationship_obj.start == table and not incoming:
                relationships.append(relationship_obj)

        return relationships

    def impute_null_value_and_replace_categorical_value(
            self, table, table_data, table_meta_data, relevant_attributes, max_distinct_vals):
        """
        Impute null value and replace categorical value
        * Refactored(Extract Function) from prepare_single_table [kihyuk-nam:2022.08.17]
        :return: table_data, table_meta_data, relevant_attributes, del_cat_attributes
        """

        # logger.info("Preparing categorical values and null values for table {}".format(table))
        table_meta_data['categorical_columns_dict'] = {}
        table_meta_data['null_values_column'] = []
        del_cat_attributes = []

        for rel_attribute in relevant_attributes:

            attribute = table + '.' + rel_attribute

            # categorical value
            if table_data.dtypes[attribute] == object:

                # logger.debug("\t\tPreparing categorical values for column {}".format(rel_attribute))

                distinct_vals = table_data[attribute].unique()

                if len(distinct_vals) > max_distinct_vals:
                    del_cat_attributes.append(rel_attribute)
                    # logger.info("Ignoring column {} for table {} because "
                    #             "there are too many categorical values".format(rel_attribute, table))
                # all values nan does not provide any information
                elif not table_data[attribute].notna().any():
                    del_cat_attributes.append(rel_attribute)
                    # logger.info(
                    #     "Ignoring column {} for table {} because all values are nan".format(rel_attribute, table))
                else:
                    if not table_data[attribute].isna().any():
                        val_dict = dict(zip(distinct_vals, range(1, len(distinct_vals) + 1)))
                        val_dict[np.nan] = 0
                    else:
                        val_dict = dict(zip(distinct_vals, range(1, len(distinct_vals) + 1)))
                        val_dict[np.nan] = 0
                    table_meta_data['categorical_columns_dict'][attribute] = val_dict

                    table_data[attribute] = table_data[attribute].map(val_dict.get)
                    # because we are paranoid
                    table_data[attribute] = table_data[attribute].fillna(0)
                    # apparently slow
                    # table_data[attribute] = table_data[attribute].replace(val_dict)
                    table_meta_data['null_values_column'].append(val_dict[np.nan])

            # numerical value
            else:

                # logger.debug("\t\tPreparing numerical values for column {}".format(rel_attribute))

                # all nan values
                if not table_data[attribute].notna().any():
                    del_cat_attributes.append(rel_attribute)
                    # logger.info(
                    #     "Ignoring column {} for table {} because all values are nan".format(rel_attribute, table))
                else:
                    contains_nan = table_data[attribute].isna().any()

                    # not the best solution but works
                    unique_null_val = table_data[attribute].mean() + 0.0001
                    assert not (table_data[attribute] == unique_null_val).any()

                    table_data[attribute] = table_data[attribute].fillna(unique_null_val)
                    table_meta_data['null_values_column'].append(unique_null_val)

                    if contains_nan:
                        assert (table_data[attribute] == unique_null_val).any(), "Null value cannot be found"

        return table_data, table_meta_data, relevant_attributes, del_cat_attributes
