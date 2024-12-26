import requests


url = "http://127.0.0.1:8000/evaluate" # API URL 
data = {
    "references": [
        "I am going to school today.",
        "I enjoy playing football in the evening.",
        "My favorite subject is mathematics.",
        "I like to read books before bed."
    ],
    "inputs": [
        "I'm heading to school now.",
        "I love playing soccer in the evening.",
        "Math is my favorite subject.",
        "I prefer reading books before sleeping."
    ]
}

# Send POST request with JSON data
response = requests.post(url, json=data)

# Print the response from the FastAPI app
if response.status_code == 200:
    print("Evaluation Metrics: ")
    print(response.json())
else:
    print(f"Error: {response.status_code} - {response.text}")
