import requests

try:
    response = requests.get('https://www.google.com', timeout=5)
    response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
    print("Successfully connected to Google!")
except requests.exceptions.RequestException as e:
    print(f"Error connecting: {e}")