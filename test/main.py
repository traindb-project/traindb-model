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

import argparse
import sys
import os
import shutil
import logging
import numpy as np
import time

from data.schemas.instacart import gen_instacart_schema
from data.preparation.prepare_single_tables import prepare_all_tables
from train.ensemble_creation.naive import create_naive_all_split_ensemble
from evaluation.aqp_evaluation import evaluate_an_aqp_query

#####
# REST API
#####
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

class Target(BaseModel):
    dataset: str
    csv_path: str

app = FastAPI()
schema = None
dataset_hdf_path = 'data/files/instacart/hdf'
table_csv_path = 'data/files/instacart/csv'+'{}.csv'

@app.post("/train/")
async def app_train(target: Target):
    schema = data_preparation(target.dataset, target.csv_path)
    result = train(schema)
    return {"Created": result, "Status": "OK"}

@app.get("/estimate/")
def aqp_read(query: str, dataset: str, ensemble_location: str, show_confidence_intervals: bool):
    #if schema is None:
    #    return {"Result":"The schema is not available"}
    value = estimate(schema, dataset, query, ensemble_location, show_confidence_intervals)
    return {"Query": query, "Estimated Value": value}

@app.put("/update/")
def aqp_update(sql: str):
    return {"Query": sql, "Updated" : "OK"}

@app.post("/delete/")
def aqp_create(model: str):
    return {"Deleted": model, "Status": "OK"}

@app.on_event("shutdown")
def shutdown_event():
#    with open("log.txt", mode="a") as log:
#        log.write("Application shutdown")
    return {"Status": "Shutdown"}

@app.on_event("startup")
def startup_event():
#    with open("log.txt", mode="a") as log:
#        log.write("Application startup")
    return {"Status": "Started"}

#####
# Main Features
#####

def data_preparation(dataset, csv_path):
    """
    Prepares the learing dataset for training process
     See, deepdb/maqp.py, schema.py
    :param dataset:
    :param csv_path:
    :return:
    """

    # - setup directories
    logger.info( "Data Preparation: ")
    dataset_path = "data/files/" + dataset
    dataset_csv_path = dataset_path + "/csv/"
    dataset_hdf_path = dataset_path + "/hdf/" # XXX set global var
    logger.info(f" - Setup Directories: input_csv_path: {csv_path}, dataset_path: {dataset_path}")

    logger.info(f" - Making csv path {dataset_csv_path}")
    os.makedirs(dataset_csv_path, exist_ok=True)

    # - extract the filename from the csv_path and make a target path
    csv_target_filename = os.path.basename(csv_path)
    csv_target_path = dataset_csv_path + csv_target_filename

    # - copy the input csv file into the target path (overwrite if already exists)
    # TODO handle the case when the csv doesn't exist
    # TODO remove if exist? just like the 'hdf'?
    logger.info(f"  (Overwrite? {os.path.exists(csv_target_path)})")
    if (csv_path != csv_target_path) and not os.path.exists(csv_target_path):
        shutil.copy(csv_path, csv_target_path)

    logger.info(f" - Making SchemaGraphs from {dataset_csv_path}")
    table_csv_path = dataset_csv_path + '{}.csv' # XXX set global var
    # XXX: seems like duplicated checking...
    if dataset == 'instacart':
        schema = gen_instacart_schema(table_csv_path) # XXX set global var
    else:
        raise ValueError('Unknown dataset')
        return False

    # - test
    table = schema.table_dictionary['orders']
    logger.info(f"   orders.table_name: {table.table_name}")
    logger.info(f"   orders.table_size: {table.table_size}")
    logger.info(f"   orders.primary_key: {table.primary_key}")
    logger.info(f"   orders.csv_file_location: {table.csv_file_location}")
    logger.info(f"   orders.sample_rate: {table.sample_rate}")

    logger.info( "Data Preparation: Generate HDF")
    logger.info(f" - Generate hdf files for the given csv and save into {dataset_hdf_path}")

    # - create hdf directory
    if os.path.exists(dataset_hdf_path):
        logger.info(f" - Removing the old {dataset_hdf_path}")
        shutil.rmtree(dataset_hdf_path)
    logger.info(f" - Making new {dataset_hdf_path}")
    os.makedirs(dataset_hdf_path)

    # - prepare all tables
    #   cf. prepare_sample_hdf in join_data_preparation.py
    logger.info(f" - Prepare all tables")
    # - requires: pip install tables
    prepare_all_tables(schema, dataset_hdf_path, csv_seperator=',', max_table_data=20000000)
    logger.info(f"Metadata(HDF files) successfully created")

    return schema


def train(schema, dataset='instacart', ensemble_path='model/instances',
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
    

    logger.info(f"TRAIN RSPNs")
    if not os.path.exists(ensemble_path):
        os.makedirs(ensemble_path)

    instance_path = None # path for the learned model file (.pkl)
    if strategy == 'single': 
        logger.info(f" - learn RSPNs by 'single' strategy")
        instance_path = \
            create_naive_all_split_ensemble(schema, dataset_hdf_path, 
                                            samples_per_spn, ensemble_path,
                                            dataset, bloom_filters, 
                                            rdc_threshold, 
                                            max_rows_per_hdf_file, 
                                            post_sampling_factor,
                                            incremental_learning_rate)
    
    logger.info(f" - create instance path (if not exists): {instance_path}")
    return instance_path


def estimate(schema, dataset, query, ensemble_location, show_confidence_intervals):
    """
    Estimate the aggregation of the given query
    :param schema:
    :param query:
    :param ensemble_location:
    :param show_confidence_intervals:
    :return:
    """
    # FIXME: incorrect answer (3421083 vs 9853061) 

    if schema is None and dataset == 'instacart':
        schema = gen_instacart_schema(table_csv_path) # XXX set global var

    logger.info(f"ESTIMATE Aggregations")
    query = "SELECT COUNT(*) FROM orders" #"SELECT SUM(order_id) FROM orders"
    ensemble_location = args.ensemble_location
    show_confidence_intervals = True
    logger.info(f" - Query: {query}, Model: {ensemble_location}")
    logger.info(f" - Show Confidence Intervals: {show_confidence_intervals}")
    result = evaluate_an_aqp_query(ensemble_location, query, schema, show_confidence_intervals)
    logger.info(f"Result: {result}")
    
    return result



#####
# Setting Up and Launching the REST API
#####
np.random.seed(1)

if __name__ == '__main__':

    #####
    # ARGS - command-line options
    #      - should match the REST API, Knative interface
    # TODO: cleanup unused args
    #####
    parser = argparse.ArgumentParser()

    # ARGS.REST 
    parser.add_argument('--rest_host', default='0.0.0.0', 
                        help='IP address of the REST API')
    parser.add_argument('--rest_port', default='8000', 
                        help='port of the REST API')

    # ARGS.DATA PREPARATION
    parser.add_argument('--dataset', default='instacart', 
                        help='dataset to be learned')
    parser.add_argument('--csv_path', default='data/files/instacart/csv/orders.csv', 
                        help='csv path for the dataset specified')
    parser.add_argument('--csv_seperator', default=',') # for tpc-ds, use '|'
    parser.add_argument('--hdf_path', default='data/files/instacart/hdf', 
                        help='csv path for the dataset specified')
    parser.add_argument('--max_rows_per_hdf_file', type=int, default=20000000)
    parser.add_argument('--hdf_sample_size', type=int, default=1000000)
    parser.add_argument('--generate_hdf', action='store_true', 
                        help='prepares hdf5 files for single tables')

    # ARGS.TRAIN
    # - learn tables to create rspn ensemble, new or update
    parser.add_argument('--train', action='store_true', 
                        help='train rspns on the given dataset')
    parser.add_argument('--ensemble_strategy', default='single')
    parser.add_argument('--ensemble_path', default='model/instances')
    parser.add_argument('--pairwise_rdc_path', default=None)
    parser.add_argument('--samples_rdc_ensemble_tests', 
                        type=int, default=10000)
    parser.add_argument('--samples_per_spn', nargs='+', type=int, 
                        default=[10000000, 10000000, 2000000, 2000000],
                        help="How many samples to use for joins with n tables")
    parser.add_argument('--post_sampling_factor', nargs='+', type=int, 
                        default=[30, 30, 2, 1])
    parser.add_argument('--rdc_threshold', type=float, default=0.3,
                        help='If RDC value is smaller independence is assumed') 
    parser.add_argument('--bloom_filters', action='store_true',
                        help='Generates Bloom filters for grouping') 
    parser.add_argument('--ensemble_budget_factor', type=int, default=5)
    parser.add_argument('--ensemble_max_no_joins', type=int, default=3)
    parser.add_argument('--incremental_learning_rate', type=int, default=0)
    parser.add_argument('--incremental_condition', type=str, default=None)

    # ARGS.ESTIMATE
    # - estimate an approximate value for the given aggregation query
    parser.add_argument('--estimate', action='store_true',
                        help='query to be approximated')
    parser.add_argument('--ensemble_location', nargs='+',
                        default=['model/instances/ensemble_single_instacart_10000000.pkl'])
    parser.add_argument('--query', default='SELECT SUM(order_id) FROM orders;')

    # ARGS.CONFIGURATION
    # - set log level
    parser.add_argument('--log_level', type=int, default=logging.DEBUG)

    # ARGS.END
    args = parser.parse_args()

    #####
    # CONF - configurations for traindb-ml
    #####
    # CONF.Logging 
    # - copied from deepdb's maqp.py
    # - depends: args.log_level, args.dataset
    #
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=args.log_level,
        # [%(threadName)-12.12s]
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                "logs/{}_{}.log".format(
                    args.dataset, time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    

    #
    # CONF.RESTAPI
    #
    # launch the fast_api (/interface/dev/main.py)
    # prerequisite: pip install fastapi uvicorn
    # testing: launch browser with "http://0.0.0.0:8000" then see hello message
    #
    #os.system('uvicorn main:app --app-dir interface/dev/ --reload --host=0.0.0.0 --port=8000')
    uvicorn.run(app, host=args.rest_host, port=int(args.rest_port))

    sys.exit("Shutting down, bye bye!")

