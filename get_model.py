import torch.nn as nn
from typing import Dict

from models import *


def model_builder(config: Dict) -> nn.Module:
    model_name = config["model"]
    if model_name == "origin":
        model = MeanConcatDense(512, 512, config["num_classes"])
        return model
    elif model_name == "early":
        model = SimpleConcat(512, 512, config["num_classes"])
        return model
    elif model_name == "late":
        model = SimplieVoting(512, 512, config["num_classes"], config["weight"])
        return model
    elif model_name == "solution1":
        model = SolutionToDifferentClass(512, 512, config["num_classes"])
        return model
    elif model_name == "mymodel":
        model = MyModel(512, 512, config["num_classes"])
        return model
    elif model_name == "fusion":
        model1 = MyModel(512, 512, config["num_classes"])
        model2 = SimplieVoting(512, 512, config["num_classes"], config["weight"])
        model3 = MeanConcatDense(512, 512, config["num_classes"])
        return model1, model2, model3
    else:
        raise ValueError(f"No model named {model_name}")
