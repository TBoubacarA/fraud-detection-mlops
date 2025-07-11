FROM python:3.9

RUN pip install --upgrade pip &&     pip install mlflow psycopg2-binary

WORKDIR /mlflow
