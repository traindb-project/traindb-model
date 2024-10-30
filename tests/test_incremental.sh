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
python $home_dir/tools/TrainDBCliModelRunner.py train RSPN $home_dir/models/RSPN.py $home_dir/tests/test_dataset/instacart_small/inc_data.csv $home_dir/tests/test_dataset/instacart_small/metadata.json $home_dir/output/

echo "SELECT COUNT(*) FROM order_products"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "COUNT(*)" "" ""

echo "SELECT COUNT(*) FROM order_products GROUP BY reordered"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "COUNT(*)" "reordered" ""

echo "SELECT COUNT(*) FROM order_products GROUP BY reordered WHERE add_to_cart_order < 4"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "COUNT(*)" "reordered" "add_to_cart_order < 4"

echo "SELECT sum(reordered) FROM order_products"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "SUM(reordered)" "" ""

echo "SELECT sum(reordered) FROM order_products GROUP BY reordered"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "SUM(reordered)" "reordered" ""

echo "SELECT sum(reordered) FROM order_products GROUP BY reordered WHERE add_to_cart_order < 4"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "SUM(reordered)" "reordered" "add_to_cart_order < 4"

echo "SELECT avg(add_to_cart_order) FROM order_products"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "AVG(add_to_cart_order)" "" ""

echo "SELECT avg(add_to_cart_order) FROM order_products GROUP BY reordered"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "AVG(add_to_cart_order)" "reordered" ""

echo "SELECT avg(add_to_cart_order) FROM order_products GROUP BY reordered WHERE add_to_cart_order < 4"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "AVG(add_to_cart_order)" "reordered" "add_to_cart_order < 4"

python $home_dir/tools/TrainDBCliModelRunner.py incremental_learn RSPN $home_dir/models/RSPN.py $home_dir/tests/test_dataset/instacart_small/inc_data2.csv $home_dir/tests/test_dataset/instacart_small/metadata.json $home_dir/output/ $home_dir/output/

echo "SELECT COUNT(*) FROM order_products"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "COUNT(*)" "" ""

echo "SELECT COUNT(*) FROM order_products GROUP BY reordered"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "COUNT(*)" "reordered" ""

echo "SELECT COUNT(*) FROM order_products GROUP BY reordered WHERE add_to_cart_order < 4"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "COUNT(*)" "reordered" "add_to_cart_order < 4"

echo "SELECT sum(reordered) FROM order_products"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "SUM(reordered)" "" ""

echo "SELECT sum(reordered) FROM order_products GROUP BY reordered"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "SUM(reordered)" "reordered" ""

echo "SELECT sum(reordered) FROM order_products GROUP BY reordered WHERE add_to_cart_order < 4"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "SUM(reordered)" "reordered" "add_to_cart_order < 4"

echo "SELECT avg(add_to_cart_order) FROM order_products"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "AVG(add_to_cart_order)" "" ""

echo "SELECT avg(add_to_cart_order) FROM order_products GROUP BY reordered"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "AVG(add_to_cart_order)" "reordered" ""

echo "SELECT avg(add_to_cart_order) FROM order_products GROUP BY reordered WHERE add_to_cart_order < 4"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "AVG(add_to_cart_order)" "reordered" "add_to_cart_order < 4"

python $home_dir/tools/TrainDBCliModelRunner.py incremental_learn RSPN $home_dir/models/RSPN.py $home_dir/tests/test_dataset/instacart_small/inc_data3.csv $home_dir/tests/test_dataset/instacart_small/metadata.json $home_dir/output/ $home_dir/output/

echo "SELECT COUNT(*) FROM order_products"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "COUNT(*)" "" ""

echo "SELECT COUNT(*) FROM order_products GROUP BY reordered"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "COUNT(*)" "reordered" ""

echo "SELECT COUNT(*) FROM order_products GROUP BY reordered WHERE add_to_cart_order < 4"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "COUNT(*)" "reordered" "add_to_cart_order < 4"

echo "SELECT sum(reordered) FROM order_products"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "SUM(reordered)" "" ""

echo "SELECT sum(reordered) FROM order_products GROUP BY reordered"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "SUM(reordered)" "reordered" ""

echo "SELECT sum(reordered) FROM order_products GROUP BY reordered WHERE add_to_cart_order < 4"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "SUM(reordered)" "reordered" "add_to_cart_order < 4"

echo "SELECT avg(add_to_cart_order) FROM order_products"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "AVG(add_to_cart_order)" "" ""

echo "SELECT avg(add_to_cart_order) FROM order_products GROUP BY reordered"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "AVG(add_to_cart_order)" "reordered" ""

echo "SELECT avg(add_to_cart_order) FROM order_products GROUP BY reordered WHERE add_to_cart_order < 4"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "AVG(add_to_cart_order)" "reordered" "add_to_cart_order < 4"
