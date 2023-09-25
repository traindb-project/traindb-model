home_dir=$(dirname -- "${BASH_SOURCE-$0}")
home_dir=$(cd -- "$home_dir/.."; pwd -P)
echo "SELECT COUNT(*) FROM order_products"
python $home_dir/tools/TrainDBCliModelRunner.py infer RSPN $home_dir/models/RSPN.py $home_dir/output/ "COUNT(*)" "" ""
