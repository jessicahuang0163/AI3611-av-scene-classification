import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    classification_report,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn

from dataset import SceneDataset
from get_model import model_builder

torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser(description="evaluation")
parser.add_argument("-e", "--experiment_path", type=str, required=True)

args = parser.parse_args()

with open(os.path.join(args.experiment_path, "config.yaml"), "r") as reader:
    config = yaml.load(reader, Loader=yaml.FullLoader)

mean_std_audio = np.load(config["data"]["audio_norm"])
mean_std_video = np.load(config["data"]["video_norm"])
mean_audio = mean_std_audio["global_mean"]
std_audio = mean_std_audio["global_std"]
mean_video = mean_std_video["global_mean"]
std_video = mean_std_video["global_std"]

audio_transform = lambda x: (x - mean_audio) / std_audio
video_transform = lambda x: (x - mean_video) / std_video

tt_ds = SceneDataset(
    config["data"]["test"]["audio_feature"],
    config["data"]["test"]["video_feature"],
    audio_transform,
    video_transform,
)
config["data"]["dataloader_args"]["batch_size"] = 1
tt_dataloader = DataLoader(tt_ds, shuffle=False, **config["data"]["dataloader_args"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config["model"] == "fusion":
    model1, model2, model3 = model_builder(config)
    model1 = model1.to(device).eval()
    model2 = model2.to(device).eval()
    model3 = model3.to(device).eval()

else:
    model = model_builder(config=config)
    model = model.to(device).eval()


if config["model"] == "fusion":
    checkpoint = torch.load(os.path.join(args.experiment_path, "best_model.pth"), "cpu")
    model1.load_state_dict(checkpoint["model1"])
    model2.load_state_dict(checkpoint["model2"])
    model3.load_state_dict(checkpoint["model3"])
else:
    model.load_state_dict(
        torch.load(os.path.join(args.experiment_path, "best_model.pt"), "cpu")
    )

targets = []
probs = []
preds = []
aids = []

with torch.no_grad():
    tt_dataloader = tqdm(tt_dataloader)
    for batch_idx, batch in enumerate(tt_dataloader):
        audio_feat = batch["audio_feat"].to(device)
        video_feat = batch["video_feat"].to(device)
        target = batch["target"].to(device)
        if config["model"] == "fusion":
            logit = (
                config["proportion"][0] * model1(audio_feat, video_feat)
                + config["proportion"][1] * model2(audio_feat, video_feat)
                + config["proportion"][2] * model3(audio_feat, video_feat)
            )
        else:
            logit = model(audio_feat, video_feat)
        pred = torch.argmax(logit, 1)
        targets.append(target.cpu().numpy())
        probs.append(torch.softmax(logit, 1).cpu().numpy())
        preds.append(pred.cpu().numpy())
        aids.append(np.array(batch["aid"]))


targets = np.concatenate(targets, axis=0)
preds = np.concatenate(preds, axis=0)
probs = np.concatenate(probs, axis=0)
aids = np.concatenate(aids, axis=0)

writer = open(os.path.join(args.experiment_path, "result.txt"), "w")
cm = confusion_matrix(targets, preds)
keys = [
    "airport",
    "bus",
    "metro",
    "metro_station",
    "park",
    "public_square",
    "shopping_mall",
    "street_pedestrian",
    "street_traffic",
    "tram",
]

scenes_pred = [keys[pred] for pred in preds]
scenes_label = [keys[target] for target in targets]
pred_dict = {"aid": aids, "scene_pred": scenes_pred, "scene_label": scenes_label}
for idx, key in enumerate(keys):
    pred_dict[key] = probs[:, idx]
pd.DataFrame(pred_dict).to_csv(
    os.path.join(args.experiment_path, "prediction.csv"),
    index=False,
    sep="\t",
    float_format="%.3f",
)


print(classification_report(targets, preds, target_names=keys), file=writer)

df_cm = pd.DataFrame(
    cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], index=keys, columns=keys
)
plt.figure(figsize=(15, 12))
sn.heatmap(df_cm, annot=True)
plt.savefig(os.path.join(args.experiment_path, "cm.png"))

acc = accuracy_score(targets, preds)
print("  ", file=writer)
print(f"accuracy: {acc:.3f}", file=writer)
logloss = log_loss(targets, probs)
print(f"overall log loss: {logloss:.3f}", file=writer)
