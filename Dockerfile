FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PORT=8000
# ---- Hugging Face cache location ----
ENV HF_HOME=/hf_cache
ENV TRANSFORMERS_CACHE=/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    unixodbc \
    unixodbc-dev \
    ca-certificates

RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg \
    && install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/ \
    && curl https://packages.microsoft.com/config/debian/11/prod.list \
    > /etc/apt/sources.list.d/mssql-release.list \
    && rm microsoft.gpg

RUN apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
    msodbcsql18

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

COPY . .

RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
