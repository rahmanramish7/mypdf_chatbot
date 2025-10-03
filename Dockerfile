FROM python:3.11-slim

WORKDIR /app

# Minimal system deps (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# pip hygiene
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# (Optional) Pre-cache the embedding model to speed up first request.
# Comment this out if you want the fastest builds instead.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"

COPY streamlit_app.py /app/streamlit_app.py

# Render provides $PORT. Streamlit must bind 0.0.0.0 and that port.
ENV PORT=10000
CMD ["sh", "-c", "streamlit run /app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0"]
