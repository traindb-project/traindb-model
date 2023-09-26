home_dir=$(dirname -- "${BASH_SOURCE-$0}")
home_dir=$(cd -- "$home_dir/.."; pwd -P)

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
