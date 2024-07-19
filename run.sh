#!/bin/bash
set -e

# Check if all three arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 [path_to_databases] [path_to_questions_json] [predicted_sql_file]"
    echo "Example: $0 path/to/test_databases path/to/test.json path/to/predicted_sql.json"
    exit 1
fi

# OPENAI_API_KEY must be defined in the environment
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY is required"
    exit 1
fi

# Input arguments
DB_PATH="$1"
QUESTIONS_FILE="$2"
OUTPUT_FILE="$3"
DESCRIBE_MODEL="claude-3.5-sonnet"
LLAMA_MODEL="llama3-output-input"
MODEL="gpt-4o"

# Intermediate outputs
METADATA_FILE=/data/db_metadata.json
PREDICTIONS_FILE=/data/intermediate_predictions.json

export OPENAI_API_BASE="https://arcwisedata--litellm-proxy-web.modal.run/v1"

echo "Generating DB schemas and metadata..."
python -m arcwise.generate_db_metadata \
  --db-path "$DB_PATH" \
  --output-file "$METADATA_FILE" \
  --model "$DESCRIBE_MODEL"

echo "Generating column & output predictions..."
python -m arcwise.llama_predict \
  --questions-file "$QUESTIONS_FILE" \
  --metadata-file "$METADATA_FILE" \
  --output-file "$PREDICTIONS_FILE" \
  --model "$LLAMA_MODEL" --concurrency 10

echo "Running Arcwise agent..."
python -m arcwise.agent.main \
  --db-path "$DB_PATH" \
  --metadata-file "$METADATA_FILE" \
  --questions-file "$PREDICTIONS_FILE" \
  --model "$MODEL" \
  --concurrency 10 \
  --output-file "$OUTPUT_FILE"

chmod a+rw "$OUTPUT_FILE"
