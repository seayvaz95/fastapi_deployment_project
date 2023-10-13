"""
Test the live prediction endpoint on Render
"""
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

example = {
  "age": 27,
  "workclass": "State-gov",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Never-married",
  "occupation": "Tech-support",
  "relationship": "Unmarried",
  "race": "White",
  "sex": "Female",
  "capital_gain": 2000,
  "capital_loss": 0,
  "hours_per_week": 35,
  "native_country": "United-States"
}

app_url = "https://fastapi-deployment-project.onrender.com/predict-income"

r = requests.post(app_url, json=example)
assert r.status_code == 200

logging.info("Testing Render app")
logging.info(f"Status code: {r.status_code}")
logging.info(f"Response body: {r.json()}")