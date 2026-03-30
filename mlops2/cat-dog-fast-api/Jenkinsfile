pipeline {
    agent any
    
    stages {
        stage("Install Dependencies") {
            steps {
                sh """
                python -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                """
            }
        }
        stage('Build Image') {
            steps {
                sh """
                . venv/bin/activate
                docker build -t deepakkumarkhatri/cat-dog-fast-api:latest .
                """
            }
        }
        stage('Push Image') {
            steps {
                withCredentials([usernamePassword(
                credentialsId: 'dockerhub-creds',
                usernameVariable: 'DOCKER_USERNAME',
                passwordVariable: 'DOCKER_PASSWORD'
                )]) {
                sh '''
                    echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
                    docker push deepakkumarkhatri/cat-dog-fast-api:latest
                '''
                }
            }
        }
        // stage('Pull Image') {
        //     steps {
        //         sh 'docker pull deepakkumarkhatri/cat-dog-fast-api:latest'
        //     }
        // }

        // stage('Deploy to Kubernetes') {
        //     steps {
        //         sh 'kubectl apply -f k8s/deployment.yaml'
        //         sh 'kubectl apply -f k8s/service.yaml'
        //     }
        // }
    }   
}
