
#
#
# data[:, :4] = data[:, :4] / rescale_xy
import os

import torch
import argparse
import numpy as np

from loguru import logger
from tqdm import tqdm

from torch import optim
from torch.nn import CrossEntropyLoss


