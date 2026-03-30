## A Volume is created called as "mlruns" which will be shared across projects.

## To run the Docker Image, following is the command - 

* docker build -t jenkins-server .
* docker run -d --name jenkins-server -p 8080:8080 -p 50000:50000 -v jenkins_home:/var/jenkins_home -v "C:/DEEPAK/BITS_AI_ML/MiscBITS/MLOpsProject/jenkins-server:/workspace" -v mlruns2:/mlruns2 jenkins-server


## Installing Docker Deamon in Jenkins server

* docker exec -it jenkins-server bash
* apt-get update
* apt-get install -y docker.io
* docker version
