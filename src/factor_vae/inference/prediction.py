import logging


import pandas as pd
import torch
from tqdm import tqdm

import torch.nn as nn

_LOG = logging.getLogger(__name__)


@torch.no_grad()
def generate_prediction_scores(
    model: nn.Module,
    test_dataloader,
    test_dataset,
    seq_len: int,
    *,
    logger: logging.Logger | None = None,
):
    log = logger or _LOG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.debug("Inference device: %s", device)
    model.to(device)

    model.eval()
    ls = []

    with tqdm(total=len(test_dataloader)) as pbar:
        for char_with_label, _ in test_dataloader:
            char = char_with_label[:, :, :-1].to(device)
            if char.shape[1] != seq_len:
                log.warning("Unexpected seq length %s (expected %s)", char.shape, seq_len)
                continue
            predictions = model.prediction(char.float())
            ls.append(predictions.detach().cpu())
            pbar.update(1)

    ls = torch.cat(ls, dim=0)
    multi_index = pd.MultiIndex.from_tuples(
        test_dataset.sampler.get_index(), names=["datetime", "instrument"]
    )
    ls = pd.DataFrame(ls.numpy(), index=multi_index, columns=["score"])
    return ls
