# type: ignore
from datetime import datetime
import os
import subprocess

import modal

app = modal.App("bird-run")

# Carbon copy of ./Dockerfile (sadly from_dockerfile doesn't work)
app_image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.5.4",
        setup_dockerfile_commands=[
            "RUN ln -s /usr/bin/python3 /usr/bin/python",  # Modal requires `python`
            "ENTRYPOINT []",  # Reset entrypoint for Modal to work
        ],
    )
    .apt_install("bash", "curl", "make", "git", "gcc", "g++", "sqlite3")
    .poetry_install_from_file("pyproject.toml")
    .workdir("/app")
    .copy_local_file("sqlite3", "/usr/bin/sqlite3")
    .run_commands("python -c \"import duckdb; duckdb.execute('INSTALL sqlite')\"")
    .copy_local_dir("arcwise", "arcwise")
    .copy_local_file("run.sh")
    .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained", HF_HUB_ENABLE_HF_TRANSFER="1"))
)

# For augmented litellm routing configs
if os.path.exists(".litellm.json"):
    app_image = app_image.copy_local_file(".litellm.json")

bird_volume = modal.Volume.from_name("bird-data")


@app.function(
    image=app_image,
    timeout=86400,
    gpu="a100-80gb:1",
    volumes={
        "/bird": bird_volume,
        "/runs": modal.Volume.from_name("runs-vol"),
        "/pretrained": modal.Volume.from_name("pretrained-vol"),
    },
    secrets=[modal.Secret.from_dotenv()],
)
def main(
    db_path: str,
    questions_file: str,
    resume_run: str | None = None,
    agent_concurrency: int = 3,
):
    run_name = resume_run or "run-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = f"/bird/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    print("Saving output to:", run_dir)

    env = os.environ.copy()
    env["AGENT_CONCURRENCY"] = str(agent_concurrency)
    env["OUTPUT_DIR"] = run_dir
    if resume_run:
        env["RESUME_RUN"] = "1"

    proc = subprocess.Popen(
        ["/app/run.sh", db_path, questions_file, run_dir + "/predict_dev.json"],
        env=env,
    )
    proc.wait()

    bird_volume.commit()


@app.function(
    image=app_image,
    timeout=3600,
    gpu="h100:1",
    volumes={
        "/bird": bird_volume,
        "/runs": modal.Volume.from_name("runs-vol"),
        "/pretrained": modal.Volume.from_name("pretrained-vol"),
    },
    secrets=[modal.Secret.from_dotenv()],
)
def schema_predict(
    questions_file: str,
    metadata_file: str,
    output_file: str,
    model: str,
    max_model_len: int = 9216,
    embedding_model: str = "azure/text-embedding-3-large",
):
    import arcwise.llama_predict as llama_predict

    llama_predict.main.callback(
        questions_file=questions_file,
        output_file=output_file,
        metadata_file=metadata_file,
        model=model,
        max_model_len=max_model_len,
        embedding_model=embedding_model,
    )
