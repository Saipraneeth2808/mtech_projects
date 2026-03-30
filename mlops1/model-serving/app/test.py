# import requests

# url = "http://localhost:5000/get-artifact"
# params = {
#     "run_uuid": "b502aea8cb454bc0aca3627c2b5d14d0",
#     "path": "logRegModel.pkl"
# }

# response = requests.get(url, params=params, stream=True)
# response.raise_for_status()

# with open("app/downloadedModel.pkl", "wb") as f:
#     for chunk in response.iter_content(chunk_size=8192):
#         if chunk:
#             f.write(chunk)

# print("File saved as downloadedModel.pkl")

# import os
# import joblib
# import pandas as pd

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print("BASE DIR ................. " + BASE_DIR)
# MODEL_PATH = os.path.join(BASE_DIR, "downloadedModel.pkl")
# model = joblib.load(MODEL_PATH)

# data = {
#     "age": 54,
#     "sex": 1,
#     "cp": 2,
#     "trestbps": 180,
#     "chol": 246,
#     "fbs": 0,
#     "restecg": 1,
#     "thalach": 150,
#     "exang": 0,
#     "oldpeak": 4.5,
#     "slope": 1,
#     "ca": 0,
#     "thal": 2
# }

# X = pd.DataFrame([data])
# pred = model.predict(X)[0]
# proba = model.predict_proba(X)[0]

# print(pred)
# print(proba)