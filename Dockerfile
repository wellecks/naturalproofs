FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY naturalproofs/ naturalproofs/
COPY setup.py setup.py
RUN (cd naturalproofs) && (python setup.py develop)

ENV PYTHONPATH=.
