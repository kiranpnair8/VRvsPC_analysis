import numpy as np
import torch

@torch.no_grad()
def collect_scores_labels_head(head, loader, device, use_claimed_id: bool):
    head.eval()
    all_scores = []
    all_labels = []

    for z, y, cid in loader:
        z = z.to(device)
        cid = cid.to(device)

        logits = head(z, cid if use_claimed_id else None).view(-1)
        scores = torch.sigmoid(logits).detach().cpu().numpy()

        all_scores.append(scores)
        all_labels.append(y.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return scores, labels


