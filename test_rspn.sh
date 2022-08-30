#!/usr/bin/env bash

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# included in all the traindb scripts with source command
# should not be executed directly
# also should not be passed any arguments, since we need original $*

home_dir=$(dirname -- "${BASH_SOURCE-$0}")
home_dir=$(cd -- "$bin"; pwd -P)

PYTHONPATH=$home_dir/models:$PYTHONPATH python $home_dir/tools/TrainDBCliModelRunner.py train RSPN $home_dir/models/RSPN.py $home_dir/tools/test_dataset/instacart_small/data.csv $home_dir/tools/test_dataset/instacart_small/metadata.json $home_dir/output/

PYTHONPATH=$home_dir/models:$PYTHONPATH python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "SELECT COUNT(*) FROM order_products"
