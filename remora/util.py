import torch
import os


def save_checkpoint(state, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    filename = os.path.join(
        out_path, "%s_%s.tar" % (state["model"], state["epoch"])
    )
    torch.save(state, filename)
