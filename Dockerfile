FROM ghcr.io/meta-pytorch/openenv-base:latest

# Install git (required)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy all code
COPY . /app

# Environment settings
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]