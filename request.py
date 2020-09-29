import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'id credit':400})

print(r.json())