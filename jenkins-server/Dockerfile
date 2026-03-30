FROM jenkins/jenkins:lts-jdk17

USER root

# ---------------------------
# Install system dependencies (IMPORTANT)
# ---------------------------
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    ca-certificates \
    build-essential \
    gnupg \
    lsb-release \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    libncursesw5-dev \
    liblzma-dev \
    tk-dev \
    uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Install Python 3.10.19
# ---------------------------
RUN wget https://www.python.org/ftp/python/3.10.19/Python-3.10.19.tgz && \
    tar -xzf Python-3.10.19.tgz && \
    cd Python-3.10.19 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd / && \
    rm -rf Python-3.10.19 Python-3.10.19.tgz

# Set Python defaults
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1

# ---------------------------
# Install pip
# ---------------------------
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# ---------------------------
# Install Docker CLI
# ---------------------------
RUN curl -fsSL https://get.docker.com | sh

# COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# RUN chmod +x /usr/local/bin/entrypoint.sh

# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Allow Jenkins user to access Docker
RUN usermod -aG docker jenkins

RUN git config --system --add safe.directory '*'

VOLUME /mlruns2

