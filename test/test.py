from model.types.RSPN import RSPN

app = RSPN()
model_path = app.train('instacart', 
                       '/home/nam/Projects/datasets/instacart/orders.csv', 
                       'data/files/', 
                       'model/instances')
print(model_path)
query = 'SELECT COUNT(*) FROM orders WHERE order_dow >= 2'
print(query)
result = app.estimate(query, 'instacart', app.table_csv_path, model_path, True) 
print(result)

'''
import requests
#url = 'http://0.0.0.0:8000/train/instacart?csv_path=~%2FProjects%2Fdatasets%2Finstacart%2Forders.csv'
url = 'http://0.0.0.0:8000/train/'
model_name = 'instacart'
dataset_path = '~/Projects/datasets/instacart/orders.csv'
payload = {'dataset':model_name, 'csv_path':dataset_path}
response = requests.post(url, json=payload)
result = response.json()
print(response.status_code)
print(result)
#url = 'http://0.0.0.0:8000/estimate/SELECT%20COUNT%28%2A%29%20FROM%20orders?dataset=instacart&ensemble_location=model%2Finstances%2Fensemble_single_instacart_10000000.pkl&show_confidence_intervals=true'
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
'''
