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

import json
import jaydebeapi
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters
import pandas as pd
from TrainDBBaseModelRunner import TrainDBModelRunner

class TrainDBPy4JModelRunner(TrainDBModelRunner):

  class Java:
    implements = [ "traindb.engine.TrainDBModelRunner" ]

  def init(self, java_port, python_port):
    self.gateway = JavaGateway(
        gateway_parameters = GatewayParameters(port=java_port),
        callback_server_parameters = CallbackServerParameters(port=python_port),
        python_server_entry_point = self)

  def connect(self, driver_class_name, url, user, password, jdbc_jar_path):
    self.conn = jaydebeapi.connect(
        driver_class_name, url, [ user, password ], jdbc_jar_path)

  def trainModel(self, sql_training_data, modeltype_class, modeltype_path, training_metadata, model_path, args=[], kwargs={}):
    curs = self.conn.cursor()
    curs.execute(sql_training_data)
    header = [desc[0] for desc in curs.description]
    data = pd.DataFrame(curs.fetchall(), columns=header)
    metadata = json.loads(training_metadata)

    model, train_info = super()._train(modeltype_class, modeltype_path, data, metadata, args, kwargs)
    model.save(model_path)

  def generateSynopsis(self, modeltype_class, modeltype_path, model_path, row_count, output_file):
    syn_data = super()._synthesize(modeltype_class, modeltype_path, model_path, row_count)
    syn_data.to_csv(output_file, index=False)

  def listHyperparameters(self, modeltype_class, modeltype_path):
    hyperparams_info = super()._hyperparams(modeltype_class, modeltype_path)
    return json.dumps(hyperparams_info)

import argparse

root_parser = argparse.ArgumentParser(description='TrainDB Model Runner')
root_parser.add_argument('java_port', type=int)
root_parser.add_argument('python_port', type=int)
args = root_parser.parse_args()

runner = TrainDBPy4JModelRunner()
runner.init(args.java_port, args.python_port)
