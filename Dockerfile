FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        openssl \
        curl \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------
COPY requirements.txt .

RUN python -m pip install --upgrade \
        pip \
        setuptools \
        wheel

RUN pip install --no-cache-dir \
    -r requirements.txt

# ------------------------------------------------------------
# Application
# ------------------------------------------------------------
COPY detection ./detection
COPY models ./models
COPY preprocessing ./preprocessing
COPY utils ./utils
COPY website ./website
COPY mlops ./mlops
COPY outputs ./outputs
COPY run_pipeline.py .

EXPOSE 5000

CMD ["python", "website/app.py"]