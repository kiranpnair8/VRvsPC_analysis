from transformers import AutoModel

out_dir = "./models"

model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, torch_dtype="auto")
pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, torch_dtype="auto")

model.save_pretrained(f"{out_dir}/reve-base")
pos_bank.save_pretrained(f"{out_dir}/reve-positions")

print("Saved REVE locally under ./models/")
