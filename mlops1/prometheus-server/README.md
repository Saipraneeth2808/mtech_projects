## A Volume is created called as "mlruns" which will be shared across projects.

## To run the Docker Image, following is the command - 

* docker build -t prometheus-server .
* docker run -d --name prometheus-server -p 9090:9090 prometheus-server
