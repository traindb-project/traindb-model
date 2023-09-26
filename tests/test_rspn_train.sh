home_dir=$(dirname -- "${BASH_SOURCE-$0}")
home_dir=$(cd -- "$home_dir/.."; pwd -P)

mkdir -p $home_dir/output
python $home_dir/tools/TrainDBCliModelRunner.py train RSPN $home_dir/models/RSPN.py $home_dir/tests/test_dataset/instacart_small/data.csv $home_dir/tests/test_dataset/instacart_small/metadata.json $home_dir/output/

