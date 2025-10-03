FROM python:3.11-slim

WORKDIR /app

# Minimal system deps (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# pip hygiene
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# where to cache HF/SBERT models at runtime
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# ❌ removed the line that downloads the model during build
# RUN python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"

COPY streamlit_app.py /app/streamlit_app.py

# Render provides $PORT. Streamlit must bind 0.0.0.0 and that port.
ENV PORT=10000
CMD ["sh", "-c", "streamlit run /app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0"]
