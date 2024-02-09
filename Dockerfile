FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04   

# Clone the llm-research repository
RUN git clone https://github.com/petavue/llm-research.git

# Set the working directory
WORKDIR /llm-research/benchmark_scripts

RUN apt install --yes --no-install-recommends vim

RUN pip uninstall traitlets
RUN pip install traitlets==5.9.0

RUN pip install --no-cache-dir xformers==0.0.22 \
    && pip install --no-cache-dir -r requirements.txt