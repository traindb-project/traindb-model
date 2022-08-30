# traindb-ml
Remote ML Model Serving Component for TrainDB

## Environment
Python 3.8 on Ubuntu 20.04

## Setting up
1. Download the codes.
```
# git clone https://github.com/traindb-project/traindb-ml.git
# cd traindb-ml
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
(venv) # pip install numpy pandas tables spflow, sqlparse, psycopg2, 
// If you want to use the Fast API, install the followings as well
(venv) # pip install fastapi uvicorn requests
// Here 'pip' means 'pip3', and it's the same as the following:
// (venv) # pip install -r requirements.txt
```
## Test non-REST version
1. Execute the test-model.py
```
(venv) # python3 test.py
(some warnings...)
model/instances/ensemble_single_instacart_10000000.pkl
SELECT COUNT(*) FROM orders WHERE order_dow >= 2
((1343995.131280017, 1347515.079651983), 1345755.105466)
```
## Launching a REST API for devel/testing (using Fast API)
1. Execute the main.py. The default host address and port (http://0.0.0.0:8000) will be applied if no args specified.
```
(venv) # python3 main.py

 // For setting up your own address/port (e.g., http://127.0.0.1:8080):
 // (venv) # python3 main.py --rest_host 127.0.0.1 --rest_port 8080
```

2. Open a web browser and go to the address you specified.
(For default setting: http://0.0.0.0:8000/docs)

![rest-all](https://user-images.githubusercontent.com/24988105/186143057-fcd91ee1-3f1e-4ad0-b22d-7819c8ccc83a.png)

## Training
1. Select the '/train/{dataset}' and click 'Try it out'.
2. Input arguments and click 'execute'. 
   - dataset: name of the dataset, which will be used a prefix of the learned model name
   - csv_path: training data, which must be in the server directory
   
     (upload and remote URL are not yet supported)
   For example:

![rest-train](https://user-images.githubusercontent.com/24988105/186143267-283a060c-33d7-443c-9c7f-ba4e448e1346.png)

3. [Option] You can test it in CLI
   ```
   import requests
   url = 'http://0.0.0.0:8000/train/'
   model_name = 'instacart'
   dataset_path = '~/Projects/datasets/instacart/orders.csv'
   payload = {'dataset':model_name, 'csv_path':dataset_path}
   response = requests.post(url, json=payload)
   result = response.json()
   print(response.status_code)
   print(result['Created')
   ```

## Estimation (AQP)
1. Select the '/estimate/{dataset}' and click 'Try it out'.
2. Input arguments and click 'execute'.
   - query: an SQL statement to be approximated. e.g., SELECT COUNT(*) FROM orders
   
     (currently COUNT, SUM, AVG are supported)
     (The table name(orders) should match the csv name(orders.csv))
   - dataset: name of the dataset you learned
   - ensemble_location: location of the learned model, which must be in the server filesystem.
   
     (upload or remote URL are not supported yet)
   - show_confidence_intervals: yes/no
   
   For example:

![rest-estimate](https://user-images.githubusercontent.com/24988105/186143491-186f857c-02ff-4daf-9241-e3c53598c5da.png)

3. [Alternative] You can test it in CLI
  ```
  import requests
  url = 'http://0.0.0.0:8000/estimate/'
  query = 'SELECT COUNT(*) FROM orders'
  dataset = 'instacart'
  model_path = 'model/instances/ensemble_single_instacart_10000000.pkl'
  show_confidence_intervals = 'true'
  payload = {'query':query,
             'dataset':dataset,
             'ensemble_location':model_path,
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
