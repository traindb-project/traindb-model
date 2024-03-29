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

home_dir=$(dirname -- "${BASH_SOURCE-$0}")
home_dir=$(cd -- "$home_dir/.."; pwd -P)

mkdir -p $home_dir/output

python $home_dir/tools/TrainDBCliModelRunner.py list TableGAN $home_dir/models/TableGAN.py $home_dir/output/tablegan_hyperparams.json
python $home_dir/tools/TrainDBCliModelRunner.py list CTGAN $home_dir/models/CTGAN.py $home_dir/output/ctgan_hyperparams.json
python $home_dir/tools/TrainDBCliModelRunner.py list TVAE $home_dir/models/TVAE.py $home_dir/output/tvae_hyperparams.json
python $home_dir/tools/TrainDBCliModelRunner.py list OCTGAN $home_dir/models/OCTGAN.py $home_dir/output/octgan_hyperparams.json
