FROM vllm/vllm-openai:v0.5.2 AS base

RUN apt update && \
  apt -y install --no-install-recommends --no-install-suggests bash curl make git gcc g++ sqlite3 && \
  apt clean && \
  rm -rf /var/cache/apt/lists

WORKDIR /app

RUN pip install poetry==1.5.1
COPY pyproject.toml poetry.lock ./
RUN POETRY_VIRTUALENVS_CREATE=false POETRY_NO_INTERACTION=1 poetry install

# TODO: Remove after upgrading to vllm v0.5.3. For Mistral Nemo
COPY vllm_llama_patched.py /usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/llama.py

# Start the main server
COPY arcwise/ /app/arcwise/
COPY run.sh .
ENTRYPOINT ["./run.sh"]
