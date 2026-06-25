FROM python:3.11-slim

LABEL maintainer="redrob-ranker-ai"
LABEL description="Redrob Hackathon — Candidate Ranking System"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY run.py .
COPY submission_metadata.yaml .

# Data directories (artifacts must be mounted or copied at runtime)
RUN mkdir -p data/raw data/processed data/artifacts outputs

# Default: ranking step only (artifacts must already exist)
# Usage:
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
#     redrob-ranker python run.py \
#       --candidates data/raw/candidates.jsonl \
#       --out outputs/submission.csv
CMD ["python", "run.py", "--help"]
