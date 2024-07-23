from modal_unsloth_mistral import app, app_image, pretrained_volume, runs_volume


@app.function(
    image=app_image,
    timeout=86400,
    gpu="t4",  # Unsloth always requires CUDA
    cpu=1.0,
    memory=65536,
    volumes={
        "/pretrained": pretrained_volume,
        "/runs": runs_volume,
    },
)
def main(
    model_name: str,
    output_path: str,
):
    assert output_path.startswith("/runs"), "Destination should be in /runs"

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=None,
        load_in_4bit=False,
        device_map="cpu",
    )
    model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")
    runs_volume.commit()
