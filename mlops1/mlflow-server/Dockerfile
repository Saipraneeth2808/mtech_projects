FROM python:3.10-slim

# Set working directory
WORKDIR /mlflow

VOLUME /mlruns2

# Install MLflow
RUN pip install --no-cache-dir mlflow

# Expose MLflow port
EXPOSE 5000

# Default command to run MLflow server
CMD ["mlflow", "server", \
     "--backend-store-uri", "/mlruns2", \
     "--default-artifact-root", "/mlruns2", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--allowed-hosts", "localhost,127.0.0.1,host.docker.internal,*"]
