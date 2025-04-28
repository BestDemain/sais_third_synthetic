FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py /app/
COPY *.yaml /app/
COPY models/*.pt /app/models/

CMD ["bash", "run.sh"]