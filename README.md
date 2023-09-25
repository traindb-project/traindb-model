# traindb-model
You can train models in ```TrainDB``` using this ML model library.
These models can be used to generate synopsis data or to estimate aggregate values in approximate query processing.

## Requirements

* [TrainDB](https://github.com/traindb-project/traindb)
* Python 3.8+
* Python virtual environment manager, such as pyenv, conda (optional)
* Packages used by ML models, such as pytorch - install requirements.txt
  * Using ```pip```: pip install -r traindb-model/requirements.txt
  * Using ```conda```: conda install --file traindb-model/requirements.txt

## Install

### Download

You can download TrainDB and this model library in one step by running the following command:
```
$> git clone --recurse-submodules https://github.com/traindb-project/traindb.git
```

## Run

If you use ```traindb-model``` library with ```TrainDB```, you can run SQL-like statements via ```trsql```.
Please refer to the README file in [TrainDB](https://github.com/traindb-project/traindb).

You can also train models and generate synthetic data using the CLI model runner.
For example, you can train a model on the test dataset as follows:
```
$> python tools/TrainDBCliModelRunner.py train TableGAN models/TableGAN.py \
       tests/test_dataset/instacart_small/data.csv \
       tests/test_dataset/instacart_small/metadata.json \
       output/
epoch 1 step 50 tensor(1.1035, grad_fn=<SubBackward0>) tensor(0.7770, grad_fn=<NegBackward>) None
epoch 1 step 100 tensor(0.8791, grad_fn=<SubBackward0>) tensor(0.9682, grad_fn=<NegBackward>) None
...

$> python tools/TrainDBCliModelRunner.py synopsis TableGAN models/TableGAN.py output 1000 sample.txt
```
## Miscellaneous on RSPN
### Dependencies
```
// major packages
# pip install numpy pandas tables spflow sqlparse psycopg2 scipy torch
// tested on Python 3.8 on Ubuntu 20.04 See the requirements.txt

// Possible errors due to the spflow package:
// 1. PyQT5: use PyQT5=5.13 if the error occurs for 5.15.x
// 2. sklearn erorr: 'sklearn' is deprecated. It should be 'scikit-learn'. See the error message that contains the solution. (set env var)
```
