# Arcwise BIRD Agent

Arcwise team submission for https://bird-bench.github.io

## Instructions

Requires Docker with CUDA and an Nvidia GPU with 80GB VRAM.

For example, to run on the included `mock_dataset`:

```bash
docker build . -t arcwise-bird

# Provided Azure OpenAI key
AZURE_API_KEY=xxxxxxxxx
# Replace with path to actual data
BIRD_PATH="$(pwd)/mock_dataset"

# Arguments: {databases_dir} {questions_file} {output_file}
# (May need to add `--runtime nvidia` on some systems.)
docker run --gpus all -v "$BIRD_PATH":/data -e AZURE_API_KEY arcwise-bird \
    /data/databases /data/questions.json /data/predict_mock.json
```

The final SQL predictions will be written to the 3rd argument.

To verify with the BIRD official evaluation script:

```
poetry run python -u ./bird_evaluation/src/evaluation.py \
  --predicted_sql_path ./mock_dataset/ --ground_truth_path ./mock_dataset/ \
  --db_root_path ./mock_dataset/databases/ --data_mode mock \
  --diff_json_path ./mock_dataset/questions.json
```

## Development

`dev_databases` / `train_databases` must be downloaded externally.

```bash
poetry install
poetry shell

# Generate db metadata
python -m arcwise.generate_db_metadata \
  --db-path ./mock_dataset/databases \
  --output-file ./mock_dataset/db_metadata.json

# augment with input/output schema hints (can use --model gpt4o)
python -m arcwise.llama_predict \
  --metadata-file ./mock_dataset/db_metadata.json \
  --questions-file ./mock_dataset/questions.json \
  --output-file ./mock_dataset/intermediate_predictions.json \
  --model arcwise/bird-mistral-nemo \
  --embedding-model azure/text-embedding-3-large

# run agent
python -m arcwise.agent.main \
  --db-path ./mock_dataset/databases \
  --metadata-file ./mock_dataset/db_metadata.json \
  --questions-file ./mock_dataset/intermediate_predictions.json \
  --output-file ./mock_dataset/predict_mock.json \
  --concurrency 4
```
