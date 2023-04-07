FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04
USER root
WORKDIR /code
COPY req.txt .
RUN apt-get update && apt-get install -y gcc g++
RUN pip install --upgrade cython
RUN pip install --no-cache-dir --user -r req.txt
RUN pip install --no-cache-dir --no-deps --user ruclip
COPY . .
CMD ["python", "app.py"]