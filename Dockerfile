FROM python:3.13

EXPOSE 8501
WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY .streamlit/* .streamlit/
COPY data/*.json data/

RUN pip install -r requirements.txt

HEALTHCHECK CMD curl --fail http://localhost:8501/healthz

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]