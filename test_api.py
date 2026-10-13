
import requests

url = "http://127.0.0.1:5000/predict"

transaction = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 10000,
    "oldbalanceOrg": 10000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 10000
}

response = requests.post(url, json=transaction)

print("Status code:", response.status_code)
print("Response:", response.json())
