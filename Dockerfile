FROM vllm/vllm-openai:v0.5.4 AS base

RUN apt update && \
  apt -y install --no-install-recommends --no-install-suggests bash curl make git gcc g++ sqlite3 && \
  apt clean && \
  rm -rf /var/cache/apt/lists

WORKDIR /app

RUN pip install poetry==1.5.1
COPY pyproject.toml poetry.lock ./
RUN POETRY_VIRTUALENVS_CREATE=false POETRY_NO_INTERACTION=1 poetry install

# Start the main server
COPY arcwise/ /app/arcwise/
COPY run.sh .
ENTRYPOINT ["./run.sh"]
