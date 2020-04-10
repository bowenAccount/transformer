import torch
import os

from H_parse import H_parse



parse=H_parse()

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if parse.device=="cpu":
    DEVICE=torch.device("cpu")

