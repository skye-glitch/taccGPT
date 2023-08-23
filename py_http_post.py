import requests

res = requests.post(url="http://localhost:9990/record_one_qa_pair/", json={"prompt":"hello", "user": "Anonymous", "answer":"hello"})
print(res)