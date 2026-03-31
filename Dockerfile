# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Simplified runtime image for Hugging Face Spaces.

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r server/requirements.txt

ENV PYTHONPATH="/app:${PYTHONPATH:-}"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
