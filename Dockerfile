FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04   

# Clone the llm-research repository
RUN git clone https://github.com/petavue/llm-research.git

# Run system updates and clean up
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && \
    apt install --yes --no-install-recommends vim && \
    rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y traitlets && \ 
    pip install traitlets==5.9.0 notebook

# Set the working directory
WORKDIR /llm-research/benchmark_scripts

RUN pip install --no-cache-dir xformers==0.0.22 \
    && pip install --no-cache-dir -r requirements.txt