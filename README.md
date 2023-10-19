# traindb-model
You can train models in ```TrainDB``` using this ML model library.
These models can be used to generate synopsis data or to estimate aggregate values in approximate query processing.

## Requirements

* [TrainDB](https://github.com/traindb-project/traindb)
* Python 3.8 or 3.9
* Python virtual environment manager, such as [pyenv](https://github.com/pyenv/pyenv) (optional)
* Packages used by ML models, such as pytorch - install requirements.txt
```
$> pip install --no-deps -r requirements.txt
```

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

## For running RSPN

### Install

You need additional dependencies for running RSPN model.
First of all, you need the SPFlow.
```
pip3 install spflow
```
If you get any error during spflow (saying sklearn xxx), then first run 'pip3 install spflow==' and try again.
It should be 'scikit-learn', not 'sklearn', but it happens frequently.
If it fails due to the PyQT5, use PyQT5==5.13 instead of 5.15.x

The following packages are also required:
```
pip3 install logging networkx==2.6.3
```
The networkx that are installed by spflow should be downgraded. Otherwise, you may get the following error:
```
...
networkx error: 'ImportError: cannot import name 'from_numpy_matrix' from 'networkx''
...
```
Install sqlparse and torch if necessary (depending on your environment).
```
pip3 install sqlparse torch
```

### Run : training

The commands/interfaces are the same as mentioned above.
For example, you can test-train an rspn model as follows:
```
python tools/TrainDBCliModelRunner.py train RSPN \
       models/RSPN.py \
       tests/test_dataset/instacart_small/data.csv \
       tests/test_dataset/instacart_small/metadata.json \
       output/
```
It creates a learned model and save in the directory you specified (the 'output' in this case).
The output directory should be exist before training.

### Run : inference

Use 'infer' command and add query arguments. For example:
```
// SELECT COUNT(*) FROM order_products GROUP BY reordered WHERE add_to_cart_order < 4
python tools/TrainDBCliModelRunner.py infer RSPN models/RSPN.py output/ "COUNT(*)" "reordered" "add_to_cart_order < 4"
```

### Demo

For demo and detailed explanation, see [Colab](https://colab.research.google.com/drive/1L1LnldEuD0pkVfxqt6-ELRxkc-DfskT1?usp=sharing).

You can run the test codes directly from the GitHub codespaces. The instructions are the same as mentioned above.
