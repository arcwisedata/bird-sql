from datetime import datetime
import os
import subprocess

import modal

app = modal.App("bird-run")

# Carbon copy of ./Dockerfile (sadly from_dockerfile doesn't work)
app_image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.5.2",
        setup_dockerfile_commands=[
            "RUN ln -s /usr/bin/python3 /usr/bin/python",  # Modal requires `python`
            "ENTRYPOINT []",  # Reset entrypoint for Modal to work
        ],
    )
    .apt_install("bash", "curl", "make", "git", "gcc", "g++", "sqlite3")
    .poetry_install_from_file("pyproject.toml")
    .workdir("/app")
    # TODO: remove after vllm v0.5.3 (for Mistral Nemo)
    .copy_local_file(
        "vllm_llama_patched.py",
        "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/llama.py",
    )
    .copy_local_dir("arcwise", "arcwise")
    .copy_local_file("run.sh")
)

# For augmented litellm routing configs
if os.path.exists(".litellm.json"):
    app_image = app_image.copy_local_file(".litellm.json")

bird_volume = modal.Volume.from_name("bird-data")


@app.function(
    image=app_image,
    timeout=86400,
    gpu="h100:1",
    volumes={"/bird": bird_volume},
    secrets=[modal.Secret.from_local_environ(["AZURE_API_KEY"])],
)
def main(
    db_path: str,
    questions_file: str,
    resume_run: str | None = None,
    agent_concurrency: int = 3,
):
    run_name = resume_run or "run-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = f"/bird/{run_name}"
    os.makedirs(run_dir)
    print("Saving output to:", run_dir)

    env = os.environ.copy()
    env["AGENT_CONCURRENCY"] = str(agent_concurrency)
    if resume_run:
        env["RESUME_RUN"] = "1"

    proc = subprocess.Popen(
        ["/app/run.sh", db_path, questions_file, run_dir + "/predict_dev.json"],
        env=env,
    )
    proc.wait()

    bird_volume.commit()
