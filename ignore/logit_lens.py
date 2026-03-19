#!/usr/bin/python3


from IPython.display import clear_output
from nnsight import LanguageModel
from typing import List, Callable
import torch
import numpy as np
from IPython.display import clear_output

clear_output()

model = LanguageModel("meta-llama/CodeLlama-7B", device_map="auto", dispatch=True)


