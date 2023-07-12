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

import importlib
import sys
import os

class TrainDBModelRunner():

  def _load_module(self, modeltype_class, modeltype_path):
    modeltype_dir = os.path.dirname(os.path.abspath(modeltype_path))
    sys.path.append(modeltype_dir)
    spec = importlib.util.spec_from_file_location(modeltype_class, modeltype_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

  def _train(self, modeltype_class, modeltype_path, real_data, table_metadata, args=[], kwargs={}):
    mod = self._load_module(modeltype_class, modeltype_path)
    model = getattr(mod, modeltype_class)(*args, **table_metadata['options'])
    model.train(real_data, table_metadata)
    return model

  def _synthesize(self, modeltype_class, modeltype_path, model_path, row_count):
    mod = self._load_module(modeltype_class, modeltype_path)
    model = getattr(mod, modeltype_class)()
    model.load(model_path)
    syn_data = model.synopsis(row_count)
    return syn_data

  def _infer(self, modeltype_class, modeltype_path, model_path, agg_expr, group_by_column, where_condition):
    mod = self._load_module(modeltype_class, modeltype_path)
    model = getattr(mod, modeltype_class)()
    model.load(model_path)
    infer_results = model.infer(agg_expr, group_by_column, where_condition)
    return infer_results

  def _hyperparams(self, modeltype_class, modeltype_path):
    mod = self._load_module(modeltype_class, modeltype_path)
    modeltype = getattr(mod, modeltype_class)
    hyperparams_info = modeltype.list_hyperparameters()
    return hyperparams_info

