FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Run system updates and clean up
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && \
    apt install --yes --no-install-recommends vim && \
    rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y traitlets && \ 
    pip install traitlets==5.9.0 notebook

# Set the working directory to /
WORKDIR /

# Clone the Fooocus repository into the workspace directory
RUN git clone https://github.com/petavue/llm-research.git

# Change the working directory to /workspace/llm-research
WORKDIR /llm-research

# Install Python dependencies
# Using '--no-cache-dir' with pip to avoid use of cache
RUN pip install --no-cache-dir xformers==0.0.22 \
    && pip install --no-cache-dir -r requirements.txt

# Change the working directory to /workspace/llm-research/benchmark_scripts
WORKDIR /llm-research/benchmark_scripts
