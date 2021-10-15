# traindb-model
Using these ML models, you can train model instances in ```TrainDB```.
These model instances can be used to generate synopsis data or to estimate aggregate values in approximate query processing.

## Requirements

* [TrainDB](https://github.com/traindb-project/traindb-prototype)
* Python 3.x
* [SDGym](https://github.com/sdv-dev/SDGym)
  * Using ```pip```: pip install sdgym
  * Using ```conda```: conda install -c sdv-dev -c conda-forge sdgym

## Install

### Download

In ```$TRAINDB_PREFIX``` directory, run the following command:
```console
$ cd $TRAINDB_PREFIX
$ svn co https://github.com/traindb-project/traindb-model/trunk/models
```

## Run

### Example

You can run SQL-like statements via ```trsql``` in ```TrainDB```.
```
$ bin/trsql
sqlline> !connect jdbc:traindb:<dbms>://<host>
Enter username for jdbc:traindb:<dbms>://localhost: <username> 
Enter password for jdbc:traindb:<dbms>://localhost: <password>
0: jdbc:traindb:<dbms>://<host>> CREATE MODEL tablegan TYPE SYNOPSIS LOCAL AS 'TableGAN' in '$TRAINDB_PREFIX/models/TableGAN.py';
No rows affected (0.255 seconds)
0: jdbc:traindb:<dbms>://<host>> TRAIN MODEL tablegan INSTANCE tgan ON <schema>.<table>(<column 1>, <column 2>, ...);
epoch 1 step 50 tensor(1.1035, grad_fn=<SubBackward0>) tensor(0.7770, grad_fn=<NegBackward>) None
epoch 1 step 100 tensor(0.8791, grad_fn=<SubBackward0>) tensor(0.9682, grad_fn=<NegBackward>) None
...
0: jdbc:traindb:<dbms>://<host>> CREATE SYNOPSIS <synopsis> FROM MODEL INSTANCE tgan LIMIT <# of rows to generate>;
...
0: jdbc:traindb:<dbms>://<host>> SELECT avg(<column>) FROM <synopsis>;
```
