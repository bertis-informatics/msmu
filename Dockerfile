FROM python:3.11

ARG GIT_TAG
ENV GIT_TAG=$GIT_TAG
RUN echo "Running build for tag: $GIT_TAG"

# For building wheels or native packages if needed
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Copy project files (msmu)
COPY . .

ARG BB_USER
ARG BB_PASS
RUN git config --global url."https://${BB_USER}:${BB_PASS}@bitbucket.org/".insteadOf "https://bitbucket.org/"

RUN pip install --upgrade pip setuptools wheel requests openpyxl
RUN pip install .
