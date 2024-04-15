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

For demo and detailed explanation, see [Notebook](https://github.com/traindb-project/traindb-model/blob/main/models/rspn/rspn.ipynb).

You can run the test codes directly from the GitHub codespaces. The instructions are the same as mentioned above.
