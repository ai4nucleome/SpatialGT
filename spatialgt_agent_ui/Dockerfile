FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY spatialgt_agent_ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire SpatialGT project
COPY . /app/

# Download gene embeddings and pretrained model at build time (optional)
# Uncomment the following lines to bake models into the image:
# RUN python -c "from core.model_manager import ensure_gene_embeddings, download_pretrained; ensure_gene_embeddings(); download_pretrained()"

WORKDIR /app/spatialgt_agent_ui

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.address", "0.0.0.0", \
    "--server.port", "8501", \
    "--server.maxUploadSize", "2000"]
