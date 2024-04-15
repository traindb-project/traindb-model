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

Similarly, you can train inference models and run queries as follows:
```
python tools/TrainDBCliModelRunner.py train RSPN \
       models/RSPN.py \
       tests/test_dataset/instacart_small/data.csv \
       tests/test_dataset/instacart_small/metadata.json \
       output/

// SELECT COUNT(*) FROM order_products GROUP BY reordered WHERE add_to_cart_order < 12
python tools/TrainDBCliModelRunner.py infer RSPN models/RSPN.py output/ "COUNT(*)" "reordered" "add_to_cart_order < 12"
```

### Demo

For demo and detailed explanation, see [Notebook](https://github.com/kihyuk-nam/traindb-model/blob/main/rspn.ipynb).
You can run the test codes directly from the GitHub codespaces. The instructions are the same as mentioned above.

