# Arcwise BIRD Agent

Arcwise team submission for https://bird-bench.github.io

## Full run

On the mock dataset:

```bash
docker build . -t arcwise-bird

export AZURE_API_KEY=xxxxxxxxx
docker run --runtime nvidia --gpus all -v $(pwd)/mock_dataset:/data -e AZURE_API_KEY \
  arcwise-bird /data/databases /data/questions.json /data/predict_mock.json

# Use official evaluation script
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
