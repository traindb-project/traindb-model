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

from model.types.TrainDBBaseModel import TrainDBModel
import os
import shutil
import logging
import time

from data.schemas.instacart import gen_instacart_schema
from data.preparation.prepare_single_tables import prepare_all_tables
from train.ensemble_creation.naive import create_naive_all_split_ensemble
from evaluation.aqp_evaluation import evaluate_an_aqp_query

class RSPN(TrainDBModel):
    """SPN-based model for TrainDB"""

    def __init__(self):
        self.dataset_name = "instacart"
        self.dataset_path = None
        self.schema = None # updated after training
        self.dataset_csv_path = 'data/files/instacart/csv'
        self.dataset_hdf_path = 'data/files/instacart/hdf'
        self.table_csv_path = 'data/files/instacart/csv'+'{}.csv' # for sharing between train() and estimate()
        self.model_path = None


        # setup logger: copied from deepdb's maqp.py
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            # level=args.log_level,
            # [%(threadName)-12.12s]
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler(
                    "logs/{}_{}.log".format(
                    __name__, time.strftime("%Y%m%d-%H%M%S"))),
                logging.StreamHandler()
                ])
        self.logger = logging.getLogger(__name__)
        self.logger.info("initialization")


    def train(self, dataset, dataset_path, metadata_file, model_path):
        """
        Learn the given dataset
        :param dataset: name of model/dataset (e.g., instacart)
        :param dataset_path: location of the dataset(.csv) (e.g., ~/Projects/datasets/instacart/orders.csv)
        :param metadata_file: location of the .hdf and .csv to be generated(or copied) (e.g., data/files/)
        :param model_path: location of the model (.pkl) to be generated (e.g., model/instances)
        :returns: path of the generated model(.pkl) if suceess
        """
        self.logger.info(f"learn the {dataset} in {dataset_path}")

        self.dataset_name = dataset
        self.schema, self.dataset_hdf_path, self.table_csv_path  = \
            self.data_preparation(dataset, dataset_path, metadata_file)
        self.model_path = \
            self.generate_rspn(schema=self.schema,
                          dataset=dataset,
                          dataset_hdf_path=self.dataset_hdf_path, 
                          ensemble_path=model_path, # XXX: .pkl path
                          strategy='single') # set the rest of the args to defaults
        self.logger.info(f"generated an RSPN ensemble in {self.model_path}.")
        
        return self.model_path

    # TODO:  
    def update(self, model_path, new_data):
        """
        Update the existing model for the newly added data
        :param model_path: location of model file (.pkl)
        :param new_data:
        :returns: path of the generated model(.pkl) if suceess
        """
        self.logger.info(f"update the {model_path} for {new_data}")

    def estimate(self, query, dataset, table_csv_path, model_path, show_confidence_intervals):
        """
        Approximate the aggregation in the query
        :param query: an SQL statement with aggregations to be approximated
        :param dataset: name of model(or dataset previously learned)
        :param model_path: location of the learned model
        :param show_confidence_intervals: yes or no
        :returns: estimated value with confidence intervals
        """
        # TODO: single table --> ensemble of tables
        # TODO: query --> components

        self.logger.info("get an approximated aggregation value for {query}")
        #query = "SELECT COUNT(*) FROM orders WHERE order_dow >= 2" 
        #dataset = 'instacart' 
        #model_path = 'model/instances/ensemble_single_instacart_10000000.pkl' 
        #show_confidence_intervals = 'true'
        if dataset == 'instacart': # XXX: needs more generalization
            schema = gen_instacart_schema(table_csv_path)
        else:
            raise ValueError('Unknown dataset')
            return False

        result = self.approximate_aggregation(schema, 
                                              dataset, 
                                              query, 
                                              model_path, 
                                              show_confidence_intervals)

        return result

    def data_preparation(self, dataset, csv_path, metadata_path):
        """
        Prepares the learing dataset for training process
        It uses the 'data' directory (e.g. data/files/csv/)
        See, deepdb/maqp.py, schema.py
        :param dataset:
        :param csv_path:
        :param metadata_path: location of the hdf and csv to be generated(copied) (e.g., data/files/)
        :return schema:
        """

        # - setup directories
        self.logger.info( "Data Preparation: ")
        dataset_path = metadata_path + dataset # XXX: what if no '/' at the end
        dataset_csv_path = dataset_path + "/csv/"
        dataset_hdf_path = dataset_path + "/hdf/"
        self.dataset_csv_path = dataset_csv_path
        self.dataset_hdf_path = dataset_hdf_path # XXX: save them to the fields
        self.logger.info(f" - Setup Directories: input_csv_path: {csv_path}, dataset_path: {dataset_path}")
        self.logger.info(f" - Making csv path {dataset_csv_path}")
        os.makedirs(dataset_csv_path, exist_ok=True)

        # - extract the filename from the csv_path and make a target path
        csv_target_path = dataset_csv_path + os.path.basename(csv_path)

        # - copy the input csv file into the target path (overwrite if already exists)
        # TODO handle the case when the csv doesn't exist
        # TODO remove if exist? just like the 'hdf'?
        self.logger.info(f"  (Overwrite? {os.path.exists(csv_target_path)})")
        if (csv_path != csv_target_path) and not os.path.exists(csv_target_path):
            shutil.copy(csv_path, csv_target_path)

        self.logger.info(f" - Making SchemaGraphs from {dataset_csv_path}")
        table_csv_path = dataset_csv_path + '{}.csv'
        self.table_csv_path = table_csv_path #XXX:  save it to the field
        # TODO: set of tables

        # XXX: seems like duplicated checking...
        if dataset == 'instacart':
            schema = gen_instacart_schema(table_csv_path)
            self.schema = schema # XXX: save it to the field
        else:
            raise ValueError('Unknown dataset')
            return False

        # - check the properties
        table = schema.table_dictionary['orders']
        self.logger.info(f"   orders.table_name: {table.table_name}")
        self.logger.info(f"   orders.table_size: {table.table_size}")
        self.logger.info(f"   orders.primary_key: {table.primary_key}")
        self.logger.info(f"   orders.csv_file_location: {table.csv_file_location}")
        self.logger.info(f"   orders.sample_rate: {table.sample_rate}")

        self.logger.info( "Data Preparation: Generate HDF")
        self.logger.info(f" - Generate hdf files for the given csv and save into {dataset_hdf_path}")

        # - create hdf directory
        if os.path.exists(dataset_hdf_path):
            self.logger.info(f" - Removing the old {dataset_hdf_path}")
            shutil.rmtree(dataset_hdf_path)
        self.logger.info(f" - Making new {dataset_hdf_path}")
        os.makedirs(dataset_hdf_path)

        # - prepare all tables
        #   cf. prepare_sample_hdf in join_data_preparation.py
        self.logger.info(f" - Prepare all tables")
        # - requires: pip install tables
        prepare_all_tables(schema, dataset_hdf_path, csv_seperator=',', max_table_data=20000000)
        self.logger.info(f"Metadata(HDF files) successfully created")

        return schema, dataset_hdf_path, table_csv_path

    def generate_rspn(self, schema, dataset, 
                      dataset_hdf_path, ensemble_path='model/instances',
                      strategy='single', rdc_threshold=0.3,
                      samples_per_spn=10000000, bloom_filters=True,
                      max_rows_per_hdf_file=20000000, post_sampling_factor=30,
                      incremental_learning_rate=0, incremental_condition=''):
        # TODO: add another strategies (e.g. relationship, rdc_based)
        # TODO: update
        """
        Learn the dataset and generate an SPN representation of it
        :param schema:
        :param dataset:
        :param ensemble_path:
        :param strategy:
        :param rdc_threshold:
        :param samples_per_spn:
        :param bloom_filters:
        :param max_rows_per_hdf_file:
        :param post_sampling_factor:
        :param incremental_learning_rate:
        :param incremental_condition:
        :return:
        """

        self.logger.info(f"TRAIN RSPNs")
        if not os.path.exists(ensemble_path):
            os.makedirs(ensemble_path)

        instance_path = None # path for the learned model file (.pkl)
        if strategy == 'single':
            self.logger.info(f" - learn RSPNs by 'single' strategy")
            instance_path = \
                create_naive_all_split_ensemble(schema, dataset_hdf_path,
                                                samples_per_spn, ensemble_path,
                                                dataset, bloom_filters,
                                                rdc_threshold,
                                                max_rows_per_hdf_file,
                                                post_sampling_factor,
                                                incremental_learning_rate)

        self.logger.info(f" - create instance path (if not exists): {instance_path}")
        return instance_path

    def approximate_aggregation(self,
                                schema, 
                                dataset, 
                                query, # TODO: decompose it (ex, op, filter)
                                ensemble_location, 
                                show_confidence_intervals):
        """
        Approximate the aggregation 
        :param schema:
        :param dataset:
        :param query:
        :param ensemble_location:
        :param show_confidence_intervals:
        :return:
        """
        # FIXME: incorrect COUNT (3421083 vs 9853061) 
        # TODO: add SUM and AVG

        # XXX: seems like weird dependency...
        #if schema is None and dataset == 'instacart':
        #    schema = gen_instacart_schema(table_csv_path)
        if schema is None:
            raise ValueError('Schema object required')
            return False
            
        self.logger.info(f"ESTIMATE Aggregations: {query}")
        ensemble_location = ensemble_location
        show_confidence_intervals = True
        self.logger.info(f" - Query: {query}, Model: {ensemble_location}")
        self.logger.info(f" - Show Confidence Intervals: {show_confidence_intervals}")
        result = evaluate_an_aqp_query(ensemble_location, query, schema, show_confidence_intervals)
        self.logger.info(f"Result: {result}")

        return result

    #
    # the following inherited methods are implemented (yet)
    #

    def save(self, output_path):
        self.logger.info("save the learned model(.pkl or .pth) to the path")

    def load(self, input_path):
        self.logger.info("load the given model")

    def save(self, output_path, file_type):
        self.logger.info("save the output ")

    def transform(self, input_model, output_model):
        self.logger.info("Transform the input .pkl into the output .pth")
