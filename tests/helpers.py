import torch
import torch.nn as nn

from microgpt.model import GPT, GPTConfig


def make_model(cfg: GPTConfig, seed: int = 0, std: float = 0.3) -> GPT:
    """Random model with *confident* outputs.

    The default init (std 0.02) gives near-uniform logits, where greedy argmax can flip on
    float noise and tests would compare ties. A wider init makes distinct, stable argmaxes.
    """
    torch.manual_seed(seed)
    model = GPT(cfg)
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                mod.weight.normal_(0.0, std)
    return model.eval()
