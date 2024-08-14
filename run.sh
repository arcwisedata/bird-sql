#!/bin/bash
set -e

# Check if all three arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 [path_to_databases] [path_to_questions_json] [output_predictions_json]"
    echo "Example: $0 path/to/test_databases path/to/test.json path/to/output_sql.json"
    exit 1
fi

# Azure OpenAI configuration
if [ -z "$AZURE_API_KEY" ]; then
    echo "AZURE_API_KEY is required"
    exit 1
fi
export AZURE_API_VERSION="2024-06-01"
export AZURE_API_BASE="https://arcwise-ai-useast2.openai.azure.com"

# Input arguments
DB_PATH="$1"
QUESTIONS_FILE="$2"
OUTPUT_FILE="$3"

# Models
CUSTOM_MODEL="arcwise/bird-mistral-nemo"
OPENAI_MODEL="azure/gpt-4o"
EMBED_MODEL="azure/text-embedding-3-large"
AGENT_CONCURRENCY=${AGENT_CONCURRENCY:-3}

# Intermediate outputs
OUTPUT_DIR=${OUTPUT_DIR:-/tmp}
METADATA_FILE="$OUTPUT_DIR/db_metadata.json"
PREDICTIONS_FILE="$OUTPUT_DIR/intermediate_predictions.json"

if [ ! -f "$METADATA_FILE" ] || [ -z "$RESUME_RUN" ]; then
  echo "Generating DB schemas and metadata..."
  python3 -m arcwise.generate_db_metadata \
    --db-path "$DB_PATH" \
    --output-file "$METADATA_FILE" \
    --model "$OPENAI_MODEL"
fi

if [ ! -f "$PREDICTIONS_FILE" ] || [ -z "$RESUME_RUN" ]; then
  echo "Generating column & output predictions..."
  python3 -m arcwise.llama_predict \
    --questions-file "$QUESTIONS_FILE" \
    --metadata-file "$METADATA_FILE" \
    --output-file "$PREDICTIONS_FILE" \
    --model "$CUSTOM_MODEL" \
    --max-model-len 20000 \
    --embedding-model "$EMBED_MODEL"
fi

echo "Running Arcwise agent..."
python3 -m arcwise.agent.main \
  --db-path "$DB_PATH" \
  --metadata-file "$METADATA_FILE" \
  --questions-file "$PREDICTIONS_FILE" \
  --model "$OPENAI_MODEL" \
  --concurrency "$AGENT_CONCURRENCY" \
  --log-file "$OUTPUT_DIR/agent.log" \
  --report-file "$OUTPUT_DIR/agent_report.json" \
  --output-file "$OUTPUT_FILE"

chmod a+rw "$OUTPUT_FILE"
