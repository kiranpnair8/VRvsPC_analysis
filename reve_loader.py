import torch
from transformers import AutoModel

def load_reve_local(
    model_base_dir: str = "./models",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(
        f"{model_base_dir}/reve-base",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    pos_bank = AutoModel.from_pretrained(
        f"{model_base_dir}/reve-positions",
        trust_remote_code=True,
        torch_dtype="auto",
    )

    model.eval().to(device)
    return model, pos_bank, device
