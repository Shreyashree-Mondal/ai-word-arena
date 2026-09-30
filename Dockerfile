FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Saved profiles live in this folder. Mount a volume here to keep them across restarts:
#   docker run -v aiword-data:/app/data ...
VOLUME ["/app/data"]

EXPOSE 8501

# Never bake a key into the image. Pass it when you run it:
#   docker run -p 8501:8501 -e AIWORD_PUBLIC_MODE=1 -e GROQ_API_KEY=... aiword
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
