FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
COPY spatialgt_agent_ui/requirements.txt /app/spatialgt_agent_ui/requirements.txt
RUN pip install --no-cache-dir -r spatialgt_agent_ui/requirements.txt

# Copy the entire SpatialGT project
COPY . /app/

# Download pretrained weights at build time (optional).
# RUN python -c "from spatialgt_agent_ui.core.model_manager import download_pretrained; download_pretrained()"

WORKDIR /app/spatialgt_agent_ui

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.address", "0.0.0.0", \
    "--server.port", "8501", \
    "--server.maxUploadSize", "2000"]
