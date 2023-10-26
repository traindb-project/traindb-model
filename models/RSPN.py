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
import logging
import time

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
                 epochs=0,
                 log_dir='test_rspn_logs',
                 log_level='debug'):
        '''
        The arguments except for rdc_threshold and log_level are not being used in this class. TODO clean up
        '''
        self.columns = [] # TODO
        self.schema = None
        self.spn_ensemble = None
        self.logger = self.setup_a_logger(log_dir, log_level) # log_dir will be created under the current dir

        # fields from deepdb, see, https://github.com/DataManagementLab/deepdb-public/blob/master/maqp.py
        self.rdc_threshold = rdc_threshold
        # TODO the following fields are not being used here. clean up unnecessary ones
        self.strategy = strategy
        self.samples_per_spn = samples_per_spn
        self.bloom_filters = bloom_filters # for custom learning (see, ~/aqp_spn/custom_spflow/*)
        self.max_rows_per_hdf_file = max_rows_per_hdf_file
        self.post_sampling_factor = post_sampling_factor
        self.incremental_learning_rate = incremental_learning_rate
        self.incremental_condition = incremental_condition

    def train(self, real_data, table_metadata):
        """
        train a model real_data(.csv) and table_metadata(.json)
        TODO extend it for multi-table (currently it only takes a single table), it should be a 'for table in tables'
        see, https://github.com/kihyuk-nam/traindb-ml/blob/main/data/preparation/prepare_single_tables.py, 
             https://github.com/DataManagementLab/deepdb-public/blob/master/data_preparation/prepare_single_tables.py
        
        :param real_data: a Pandas DataFrame of training_data(.csv)
        :param table_metadata: a deserialized json object of metadata.json
        :return: None. a learned model(SPNEnsemble object) is saved in the self.spn_ensemble
        """
        self.logger.info(f"Preparing training data")
        # 1. collect table info (model_columns, categorical_columns, table_size)
        # TODO 
        columns, categoricals = self.get_columns(real_data, table_metadata) # see, models/TrainDBBaseModel.py
        real_data = real_data[columns]
        self.columns = columns
        table_size = len(real_data) # TODO cf. deepdb's sampling for huge training datasets
        # cf. https://github.com/DataManagementLab/deepdb-public/blob/master/data_preparation/join_data_preparation.py#L239,275,330
        self.logger.debug(f"- table size: {table_size}, columns: {columns}, categorical columns: {categoricals}")

        # 2. prepare schema object
        # see, https://github.com/DataManagementLab/deepdb-public/blob/master/maqp.py#L114
        # cf. gen_instacart_schema() in https://github.com/kihyuk-nam/traindb-ml/blob/main/data/schemas/instacart.py
        #     which is a variant of https://github.com/DataManagementLab/deepdb-public/tree/master/schemas/
        schema = SchemaGraph() # ensemble_compilation/graph_representation.py#L76        
        schema.add_table(Table(table_metadata['table'], attributes=columns, table_size=table_size))

        # 3. prepare rspn_table_metadata, table_set, and update real_data
        rspn_table_metadata, table_set, real_data = self.prepare_for_training(schema, real_data, table_metadata, columns)
        self.logger.debug(f"- table set: {table_set}")
        self.logger.debug(f"- real_data.columns/rspn_columns: {real_data.columns.values.tolist()}")

        # 4. join data preparation : meta_types, null_values, full_join_size, full_sample_size
        meta_types, null_values, full_join_size, full_sample_size = self.generate_join_samples(columns, categoricals, table_size)

        self.logger.info(f"Start learning")
        # 5. start training
        # see, https://github.com/DataManagementLab/deepdb-public/blob/master/ensemble_creation/naive.py
        # AQPSQPN(CombinedSPN, RSPN): https://github.com/DataManagementLab/deepdb-public/blob/master/aqp_spn/aqp_spn.py#L19
        aqp_spn = AQPSPN(meta_types, null_values, full_join_size, schema, None, full_sample_size,
                         table_set=table_set, column_names=real_data.columns.values.tolist(), table_meta_data=rspn_table_metadata)
        # TODO min_instance_slice = RATIO_MIN_INSTANCE_SLICE * min(sample_size, len(df_samples))
        aqp_spn.learn(real_data.to_numpy(), rdc_threshold=self.rdc_threshold)

        # 6. wrap up. save the learned model and the schema in the corresponding fields.
        self.schema = schema
        self.spn_ensemble = SPNEnsemble(schema)
        self.spn_ensemble.add_spn(aqp_spn)

    def save(self, output_path):
        """
        saves the learned model (spn_ensemble) as two files in the output_path
        - model.pth: using torch.save
        - spn_ensembles: using pickle (bz2 compressed) See. spn_ensemble.py#L596
        :param output_path: dir where the model files are saved
        :return: files are saved in the output_path
        """
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
        hparams.append(TrainDBModel.createHyperparameter('rdc_threshold', 'float', '0.3', 'threshold for determining correlation'))
        hparams.append(TrainDBModel.createHyperparameter('log_level', 'str', 'info', 'level of log message'))
        ''' TODO clean up args: currently the following arguments are not being used.
        hparams.append(TrainDBModel.createHyperparameter('strategy', 'str', 'single', 'type of RSPN ensemble'))
        hparams.append(TrainDBModel.createHyperparameter('sample_per_spn', 'int', '10000000', 'the number of samples per spn'))
        hparams.append(TrainDBModel.createHyperparameter('bloom_filters', 'bool', 'True', 'whether bloom_filter applies or not'))
        hparams.append(TrainDBModel.createHyperparameter('max_rows_per_hdf_file', 'int', '20000000', 'max rows per hdf file'))
        hparams.append(TrainDBModel.createHyperparameter('post_sampling_factor', 'int', '30', 'post sampling factor'))
        hparams.append(TrainDBModel.createHyperparameter('incremental_learning_rate', 'int', '0', 'init learning / incremental'))
        hparams.append(TrainDBModel.createHyperparameter('incremental_condition', 'str', '', 'predicate for incremental learning'))
        hparams.append(TrainDBModel.createHyperparameter('epochs', 'int', '0', 'the number of training epochs'))
        '''
        return hparams

    def setup_a_logger(log_dir, log_level):    
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s", # [%(threadName)-12.12s]
            handlers=[
                logging.FileHandler("test_rspn_logs/{}_{}.log".format("rspn", time.strftime("%Y%m%d-%H%M%S"))),
                logging.StreamHandler()
            ])
        logger = logging.getLogger(__name__)
        if log_level == 'debug':
            logger.setLevel(logging.DEBUG)
        elif log_level == 'info':
            logger.setLevel(logging.INFO)
        elif log_level == 'warning':
            logger.setLevel(logging.WARNING)
        elif log_level == 'error':
            logger.setLevel(logging.ERROR)
        elif log_level == 'critical':
            logger.setLevel(logging.CRITICAL)

        return logger
    
    def prepare_for_training(self, schema, real_data, table_metadata, columns):
        """
        Prepare for training arguments - create rspn_table_metadata, table_set and update real_data
        see, https://github.com/DataManagementLab/deepdb-public/blob/master/data_preparation/prepare_single_tables.py#L255
        :param schema: a SchemaGraph instance of the training data(data.csv)
        :param real_data: training data (data.csv)
        :return: rspn_table_metadata, table_set, real_data
        """
        # TODO reorganize them
        rspn_table_metadata = dict()
        table_set = set()
        for table in schema.tables:
            table_set.add(table.table_name)
            rspn_table_metadata[table.table_name] = dict()
        table_obj = schema.table_dictionary[table_metadata['table']]
        table = table_obj.table_name
        real_data.columns = [table + '.' + col for col in columns] # e.g., [order_products.reordered, order_products.add_to_cart_order]
        
        relevant_attributes = [x for x in table_obj.attributes if x not in table_obj.irrelevant_attributes] # == columns. TODO check!

        rspn_table_metadata, real_data, relevant_attributes = self.manage_functional_dependencies(
            table, table_obj, real_data, rspn_table_metadata, relevant_attributes)
        #   see, https://github.com/DataManagementLab/deepdb-public/blob/master/data_preparation/prepare_single_tables.py#L58


        # TODO rspn_table_metadata, real_data, relevant_attributes = self.add_multiplier_fields(
        #    table, table_obj, real_data, rspn_table_metadata, relevant_attributes, schema, csv_seperator=',')
        #   see, https://github.com/DataManagementLab/deepdb-public/blob/master/data_preparation/prepare_single_tables.py#L82
        #   see, https://github.com/kihyuk-nam/traindb-ml/blob/main/data/preparation/prepare_single_tables.py#L53

        rspn_table_metadata = self.save_entities_without_fk_reference(schema, table, rspn_table_metadata, real_data)
        # (e.g. orders without customers)
        #   see, https://github.com/DataManagementLab/deepdb-public/blob/master/data_preparation/prepare_single_tables.py#L126
        
        real_data, rspn_table_metadata, relevant_attributes, del_cat_attributes = \
            self.impute_null_value_and_replace_categorical_value(
                table, real_data, rspn_table_metadata, relevant_attributes, 10000)
        #   see, https://github.com/DataManagementLab/deepdb-public/blob/master/data_preparation/prepare_single_tables.py#L137
        
        # TODO do things like
        # - remove categorical columns with too many entries from relevant tables and dataframe
        # - save modified table
        # - add table parts without join partners
        #   see, https://github.com/DataManagementLab/deepdb-public/blob/master/data_preparation/prepare_single_tables.py#L200
    
        return rspn_table_metadata, table_set, real_data


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

    def add_multiplier_fields(table, table_obj, table_data, table_meta_data, 
                              relevant_attributes, schema_graph, csv_seperator):
        """
        Add multiplier fields
        :return: table_meta_data, table_data, relevant_attributes, incoming_relationships
        """
        logger.info("Preparing multipliers for table {}".format(table))
        incoming_relationships = find_relationships(schema_graph, table, incoming=True)

        for relationship_obj in incoming_relationships:
            logger.info("Preparing multiplier {} for table {}".format(relationship_obj.identifier, table))

            neighbor_table = relationship_obj.start
            neighbor_table_obj = schema_graph.table_dictionary[neighbor_table]
            neighbor_sample_rate = neighbor_table_obj.sample_rate

            left_attribute = table + '.' + relationship_obj.end_attr
            right_attribute = neighbor_table + '.' + relationship_obj.start_attr

            neighbor_table_data = read_table_csv(neighbor_table_obj, csv_seperator=csv_seperator).set_index(right_attribute,
                                                                                                            drop=False)
            table_data = table_data.set_index(left_attribute, drop=False)

            assert len(table_obj.primary_key) == 1, \
                "Currently, only single primary keys are supported for table with incoming edges"
            table_primary_key = table + '.' + table_obj.primary_key[0]
            assert table_primary_key == left_attribute, "Currently, only references to primary key are supported"

            # fix for new pandas version
            table_data.index.name = None
            neighbor_table_data.index.name = None
            muls = table_data.join(neighbor_table_data, how='left')[[table_primary_key, right_attribute]] \
                .groupby([table_primary_key]).count()

            mu_nn_col_name = relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name_nn
            mu_col_name = relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name

            muls.columns = [mu_col_name]
            # if we just have a sample of the neighbor table we assume larger multipliers
            muls[mu_col_name] = muls[mu_col_name] * 1 / neighbor_sample_rate
            muls[mu_nn_col_name] = muls[mu_col_name].replace(to_replace=0, value=1)

            table_data = table_data.join(muls)

            relevant_attributes.append(relationship_obj.multiplier_attribute_name)
            relevant_attributes.append(relationship_obj.multiplier_attribute_name_nn)

            table_meta_data['incoming_relationship_means'][relationship_obj.identifier] = table_data[mu_nn_col_name].mean()

        return table_meta_data, table_data, relevant_attributes, incoming_relationships 

    def save_entities_without_fk_reference(self, schema, table, rspn_table_metadata, real_data):
        outgoing_relationships = self.find_relationships(schema, table, incoming=False)
        for relationship_obj in outgoing_relationships:
            fk_attribute_name = table + '.' + relationship_obj.start_attr

            rspn_table_metadata[relationship_obj.identifier] = {
                'fk_attribute_name': fk_attribute_name,
                'length': real_data[fk_attribute_name].isna().sum(),
                # 'length': table_data[fk_attribute_name].isna().sum() * 1 / table_sample_rate,
                'path': None
            }
        return rspn_table_metadata
        
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

    def generate_join_samples(self, columns, categoricals, table_size):
        """
        generate join samples (which are currently dummy values)
        see, https://github.com/DataManagementLab/deepdb-public/blob/master/ensemble_creation/naive.py#L23, 31, 67, 71
            by calling generate_n_samples in ~/data_preparation/join_data_preparation.py#L239
            which can be further sampled for fast join calculation by prepare_sample_hdf(~#L619) 
        """
        # set meta_types, null_values 
        # TODO get real values
        # see, https://github.com/DataManagementLab/deepdb-public/blob/master/data_preparation/join_data_preparation.py#L256, 266, 266
        #    by calling https://github.com/DataManagementLab/deepdb-public/blob/master/data_preparation/join_data_preparation.py#L330
        meta_types = []
        null_values = []
        for col in columns:
            if col in categoricals:
                meta_types.append(MetaType.DISCRETE)
            else:
                meta_types.append(MetaType.REAL)
            null_values.append(None)

        # set full_join_size
        # TODO get a real value
        # see, https://github.com/DataManagementLab/deepdb-public/blob/master/data_preparation/join_data_preparation.py#L250, L286
        #    by calling https://github.com/DataManagementLab/deepdb-public/blob/master/data_preparation/join_data_preparation.py#L179
        full_join_size = table_size

        # set full_sample_size (which sets to full_join_size if not specified otherwise)
        # see, https://github.com/DataManagementLab/deepdb-public/blob/master/aqp_spn/aqp_spn.py#L26-L27
        full_sample_size = full_join_size

        return meta_types, null_values, full_join_size, full_sample_size