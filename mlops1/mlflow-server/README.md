## A Volume is created called as "mlruns" which will be shared across projects.

## To run the Docker Image, following is the command - 

* docker build -t mlflow-server .
* docker run -d --name mlflow -p 5000:5000 -v mlruns2:/mlruns2 mlflow-server
