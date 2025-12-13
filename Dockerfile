FROM python:3.11-slim

# system deps (fonts, tz, build tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential tini && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects the app to bind to $PORT
ENV PORT=7860
EXPOSE 7860
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860"]
