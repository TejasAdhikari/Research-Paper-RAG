FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", ".\phase4_WebUI\rag_web_app_full.py", "--server.port=8501", "--server.address=0.0.0.0"]

