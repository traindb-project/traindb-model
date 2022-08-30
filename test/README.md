# RSPN Testing

## Environment
Python 3.8 on Ubuntu 20.04

## Setting up
1. Download the codes.
```
# git clone https://github.com/traindb-project/traindb-model.git
# cd traindb-model/test
```
2. (Option) Create a virtual environment (using venv or conda).
  - For venv: (See:https://docs.python.org/3/library/venv.html)
    ```
    # python3 -m venv venv 
    # source venv/bin/activate
    (venv) #
    ```

3. Install dependencies. For example,
```
(venv) # pip install numpy pandas tables spflow, sqlparse, psycopg2
// If you want to use REST interface (Fast API), install the followings as well
(venv) # pip install fastapi uvicorn requests
// Here 'pip' means 'pip3', and it's the same as the following:
// (venv) # pip install -r requirements.txt
```
## Running Option1: TrainDBCliModelRunner
### Training
```
(venv) # python3 TrainDBCliModelRunner.py train2 RSPN model/types/RSPN.py instacart /data/instacart/orders.csv data/files/ model/instances

(some warnings...)
```
- train2: command for RSPN testing
- modeltype_class: name of the model (class) (ex, RSPN)
- modeltype_uri: path for the model class file (ex, model/types/RSPN.py)
- data_name: name(space) of the training dataset, (ex, instacart)
- data_file: path to the training dataset, /path/to/namespace/tablename.csv (ex, /data/instacart/orders.csv)
- metadata_root: root dir of the metadata(.json or .hdf) (to be modified)
- model_path: (root) path to the generated model (ex, model/instances)

### Estimation
...

## Running Option2: Instantiate RSPN
### Training and Estimation
1. Write a script. For example, test.py
```
from model.types.RSPN import RSPN

app = RSPN()

# training
model_path = app.train('instacart',
                       '/home/nam/Projects/datasets/instacart/orders.csv',
                       'data/files/',
                       'model/instances')
print(model_path)

# estimation
query = 'SELECT COUNT(*) FROM orders WHERE order_dow >= 2'
print(query)
result = app.estimate(query, 'instacart', app.table_csv_path, model_path, True)
print(result)
```
2. Check the result:
```
(venv) # python3 test.py

(some warnings...)

model/instances/ensemble1_single_instacart_10000000.pkl
SELECT COUNT(*) FROM orders WHERE order_dow >= 2
((1343995.131280017, 1347515.079651983), 1345755.105466)
```
## Running Option3: REST API (using Fast API)
1. Execute the rest.py. 
The default host address and port (http://0.0.0.0:8000) will be applied if no args specified.
```
(venv) # python3 rest.py

 // For setting up your own address/port (e.g., http://127.0.0.1:8080):
 // (venv) # python3 main.py --rest_host 127.0.0.1 --rest_port 8080
```

2. Open a web browser and go to the address you specified.
(For default setting: http://0.0.0.0:8000/docs)

![rest-all](https://user-images.githubusercontent.com/24988105/186143057-fcd91ee1-3f1e-4ad0-b22d-7819c8ccc83a.png)

### Training
1. Select the '/train/' and click 'Try it out'.
2. Input arguments and click 'execute'. 
   - dataset: name of the dataset, which will be used a prefix of the learned model name
   - csv_path: training data, which must be in the server directory
   - metadata_path: .json or .hdf file containing metadata of the input(csv)
   - model_path: location of the model to be generated
   
     (upload and remote URL are not yet supported)
   For example:
   
![rest-train](https://user-images.githubusercontent.com/24988105/187427079-87603e1f-2cfa-466e-a0ef-fae55817c177.png)

3. [Option] CLI
   - install requests package if not exists
   ```
   (venv) # pip install requests
   ```
   - write a script calling APIs. For example,
   ```
   import requests
   url = 'http://0.0.0.0:8000/train/'
   model_name = 'instacart'
   csv_path = '/datasets/instacart/orders.csv'
   metadata_path = 'data/files/'
   model_path = 'model/instances'
   payload = {'dataset':model_name, 'csv_path':dataset_path, 'metadata_path':metadata_path, 'model_path'=model_path}
   response = requests.post(url, json=payload)
   result = response.json()
   print(response.status_code)
   print(result['Created')
   ```

### Estimation (AQP)
1. Select the '/estimate/' and click 'Try it out'.
2. Input arguments and click 'execute'.
   - query: an SQL statement to be approximated. 
     
     e.g., SELECT COUNT(*) FROM orders WHERE order_dow >= 2
     
     (currently COUNT is supported. SUM and AVG will be available soon)
     
     (The table name(orders) should match the csv name(orders.csv))
     
   - dataset: name(space) of the dataset you learned
   - model_path: location of the learned model, which must be uploaded in advance in the server filesystem.
   - table_csv_path: (temporary parameter) location of the data file used for training
   - show_confidence_intervals: yes/no
   
   For example:
   
![rest-estimate](https://user-images.githubusercontent.com/24988105/187428163-f9f342a8-fe55-40df-91f9-82d4b7b7a1e8.png)


3. [Option] CLI
  ```
  import requests
  url = 'http://0.0.0.0:8000/estimate/'
  query = 'SELECT COUNT(*) FROM orders WHERE order_dow >= 2'
  dataset = 'instacart'
  table_csv_path = 'data/files/instacart/csv/orders.csv'
  model_path = 'model/instances/ensemble_single_instacart_10000000.pkl'
  show_confidence_intervals = 'true'
  payload = {'query':query,
             'dataset':dataset,
             'table_csv_path':table_csv_path,
             'model_path':model_path,
             'show_confidence_intervals':show_confidence_intervals}

  response = requests.get(url, params=payload)
  result = response.json()
  print(response.status_code)
  print(result['Estimated Value'])
  ```
## Using KubeFlow
- See [/interface](https://github.com/traindb-project/traindb-ml/tree/main/interface)

## License
This project is dual-licensed. Apache 2.0 is for traindb-ml, and MIT is for the RSPN codes from deepdb(https://github.com/DataManagementLab/deepdb-public)
