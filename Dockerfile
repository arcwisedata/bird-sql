FROM vllm/vllm-openai:v0.5.4 AS base

RUN apt update && \
  apt -y install --no-install-recommends --no-install-suggests bash curl make git gcc g++ sqlite3 unzip && \
  apt clean && \
  rm -rf /var/cache/apt/lists

WORKDIR /app

RUN pip install poetry==1.5.1
COPY pyproject.toml poetry.lock ./
RUN POETRY_VIRTUALENVS_CREATE=false POETRY_NO_INTERACTION=1 poetry install

# Manual build of sqlite3 v3.43.0
COPY sqlite3 /usr/bin/sqlite3
RUN chmod a+x /usr/bin/sqlite3
RUN sqlite3 --version

# Install duckdb
RUN curl https://github.com/duckdb/duckdb/releases/download/v1.0.0/duckdb_cli-linux-amd64.zip -L -o duckdb_cli-linux-amd64.zip && \
  unzip duckdb_cli-linux-amd64.zip && \
  rm duckdb_cli-linux-amd64.zip && \
  mv duckdb /usr/bin/duckdb && \
  chmod a+x /usr/bin/duckdb && \
  duckdb --version

# Start the main server
COPY arcwise/ /app/arcwise/
COPY run.sh .
ENTRYPOINT ["./run.sh"]
