# Use an official Python base image (choose a tag that works for your project)
FROM python:3.9-slim

# Install system dependencies (build-essential, etc.) needed for some packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy a requirements file into the container
# Create a requirements.txt with the libraries you need, for example:
# torch==2.0.0
# biopython
# matplotlib
# (plus any other libraries)
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set the working directory in the container
WORKDIR /workspace

# Copy your project code into the container
COPY . /workspace

# Optionally, expose any ports your training script might need
EXPOSE 29501

# Default command to run your training script
CMD ["python", "progressive_ddp_training_server.py"]
