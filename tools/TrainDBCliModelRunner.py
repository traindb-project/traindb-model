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
import os
from TrainDBBaseModelRunner import TrainDBModelRunner

class TrainDBCliModelRunner(TrainDBModelRunner):

  def train_model(self, modeltype_class, modeltype_path, real_data, table_metadata, model_path, args=[], kwargs={}):
    model = super()._train(modeltype_class, modeltype_path, real_data, table_metadata, args, kwargs)
    model.save(model_path)

  def incremental_learn(self, modeltype_class, modeltype_path, model_path, incremental_data, table_metadata, args=[], kwargs={}):
    model = super()._incremental_learn(modeltype_class, modeltype_path, model_path, incremental_data, table_metadata, args, kwargs)
    model.save(model_path)

  def generate_synopsis(self, modeltype_class, modeltype_path, model_path, row_count):
    syn_data = super()._synthesize(modeltype_class, modeltype_path, model_path, row_count)
    return syn_data

  def infer(self, modeltype_class, modeltype_path, model_path, agg_expr, group_by_column, where_condition):
    infer_results = super()._infer(modeltype_class, modeltype_path, model_path, agg_expr, group_by_column, where_condition)
    return infer_results

  def list_hyperparameters(self, modeltype_class, modeltype_path):
    hyperparams_info = super()._hyperparams(modeltype_class, modeltype_path)
    return json.dumps(hyperparams_info)

  def evaluate(self, real_data, synopsis_data, table_metadata):
    quality_report = super()._evaluate(real_data, synopsis_data, table_metadata)
    return quality_report

import argparse
import pandas as pd
import json
import sys
import csv
from datasets import load_dataset

def main():
  root_parser = argparse.ArgumentParser(description='TrainDB CLI Model Runner')
  subparsers = root_parser.add_subparsers(dest='cmd')
  parser_train = subparsers.add_parser('train', help='train model command')
  parser_train.add_argument('modeltype_class', type=str, help='(str) modeltype class name')
  parser_train.add_argument('modeltype_uri', type=str, help='(str) path for local model, or uri for remote model')
  parser_train.add_argument('data_file', type=str, help='(str) path to .csv data file')
  parser_train.add_argument('metadata_file', type=str, help='(str) path to .json table metadata file')
  parser_train.add_argument('model_path', type=str, help='(str) path to model')

  parser_train = subparsers.add_parser('incremental_learn', help='incremental_learn command')
  parser_train.add_argument('modeltype_class', type=str, help='(str) modeltype class name')
  parser_train.add_argument('modeltype_uri', type=str, help='(str) path for local model, or uri for remote model')
  parser_train.add_argument('data_file', type=str, help='(str) path to .csv data file')
  parser_train.add_argument('metadata_file', type=str, help='(str) path to .json table metadata file')
  parser_train.add_argument('model_path', type=str, help='(str) path to model')

  parser_synopsis = subparsers.add_parser('synopsis', help='generate synopsis command')
  parser_synopsis.add_argument('modeltype_class', type=str, help='(str) modeltype class name')
  parser_synopsis.add_argument('modeltype_uri', type=str, help='(str) path for local model, or uri for remote model')
  parser_synopsis.add_argument('model_path', type=str, help='(str) path to model')
  parser_synopsis.add_argument('row_count', type=int, help='(int) the number of rows to generate')
  parser_synopsis.add_argument('output_file', type=str, help='(str) path to save generated synopsis file')

  parser_infer = subparsers.add_parser('infer', help='inference model command')
  parser_infer.add_argument('modeltype_class', type=str, help='(str) modeltype class name')
  parser_infer.add_argument('modeltype_uri', type=str, help='(str) path for local model, or uri for remote model')
  parser_infer.add_argument('model_path', type=str, help='(str) path to model')
  parser_infer.add_argument('agg_expr', type=str, help='(str) aggregation expression')
  parser_infer.add_argument('group_by_column', type=str, help='(str) column specified in GROUP BY clause')
  parser_infer.add_argument('where_condition', type=str, help='(str) filter condition specified in WHERE clause')
  parser_infer.add_argument('output_file', type=str, nargs='?', default='', help='(str) path to save inferred query result file')

  parser_list = subparsers.add_parser('list', help='list model hyperparameters')
  parser_list.add_argument('modeltype_class', type=str, help='(str) modeltype class name')
  parser_list.add_argument('modeltype_uri', type=str, help='(str) path for local model, or uri for remote model')
  parser_list.add_argument('output_file', type=str, help='(str) path to .json model hyperparameters file')

  parser_evaluate = subparsers.add_parser('evaluate', help='evaluate synopsis command')
  parser_evaluate.add_argument('data_file', type=str, help='(str) path to .csv data file')
  parser_evaluate.add_argument('synopsis_file', type=str, help='(str) path to .csv synopsis file')
  parser_evaluate.add_argument('metadata_file', type=str, help='(str) path to .json table metadata file')
  parser_evaluate.add_argument('output_file', type=str, help='(str) path to save generated quality report file')

  args = root_parser.parse_args()
  runner = TrainDBCliModelRunner()
  if args.cmd == 'train':
    data_file = load_dataset("csv", data_files=args.data_file)
    data_file.set_format(type="pandas")
    df_train = data_file["train"][:]
    with open(args.metadata_file) as metadata_file:
      table_metadata = json.load(metadata_file)
    runner.train_model(args.modeltype_class, args.modeltype_uri, df_train, table_metadata, args.model_path)
    sys.exit(0)
  elif args.cmd == 'synopsis':
    syn_data = runner.generate_synopsis(args.modeltype_class, args.modeltype_uri, args.model_path, args.row_count)
    syn_data.to_csv(args.output_file, index=False)
    sys.exit(0)
  elif args.cmd == 'infer':
    aqp_result, confidence_interval = runner.infer(args.modeltype_class, args.modeltype_uri, args.model_path, args.agg_expr, args.group_by_column, args.where_condition)
    if args.output_file == '':
      print(aqp_result)
    else:
      with open(args.output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(aqp_result)
    sys.exit(0)
  elif args.cmd == 'list':
    json_hyperparams_info = runner.list_hyperparameters(args.modeltype_class, args.modeltype_uri)
    with open(args.output_file, 'w') as f:
      f.write(json_hyperparams_info)
    sys.exit(0)
  elif args.cmd == 'evaluate':
    orig_data = pd.read_csv(args.data_file)
    syn_data = pd.read_csv(args.synopsis_file)
    with open(args.metadata_file) as metadata_file:
      table_metadata = json.load(metadata_file)
    quality_report = runner.evaluate(orig_data, syn_data, table_metadata)
    column_shapes = quality_report.get_details(property_name='Column Shapes')
    column_shapes.to_json(args.output_file, orient='records', lines=True)

    # for visualization
    #print(quality_report.get_details(property_name='Column Shapes'))
    #fig = quality_report.get_visualization(property_name='Column Shapes')
    #fig.show()
    sys.exit(0)
  elif args.cmd == 'incremental_learn':
    data_file = pd.read_csv(args.data_file)
    with open(args.metadata_file) as metadata_file:
      table_metadata = json.load(metadata_file)  
    runner.incremental_learn(args.modeltype_class, args.modeltype_uri, args.model_path, data_file, table_metadata, args.model_path)  
    sys.exit(0)      
  else:
    root_parser.print_help()

if __name__ == "__main__":
  main()
