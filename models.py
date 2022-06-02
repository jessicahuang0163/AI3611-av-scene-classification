import torch
import torch.nn as nn


class MeanConcatDense(nn.Module):
    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
        )
        self.outputlayer = nn.Sequential(
            nn.Linear(256, 128), nn.Linear(128, self.num_classes),
        )

    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)
        audio_emb = self.audio_embed(audio_emb)

        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)

        embed = torch.cat((audio_emb, video_emb), 1)
        output = self.outputlayer(embed)
        return output


class SimpleConcat(nn.Module):
    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(audio_emb_dim + video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
        )

        self.outputlayer = nn.Sequential(
            nn.Linear(256, 128), nn.Linear(128, self.num_classes),
        )

    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)
        video_emb = video_feat.mean(1)
        embed = torch.cat((audio_emb, video_emb), 1)
        output = self.net(embed)
        output = self.outputlayer(output)
        return output


class SimplieVoting(nn.Module):
    def __init__(self, audio_emb_dim, video_emb_dim, num_classes, weight) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.audio_pred = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.Linear(128, self.num_classes),
        )
        self.video_pred = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)
        audio_pred = self.audio_pred(audio_emb)

        video_emb = video_feat.mean(1)
        video_pred = self.video_pred(video_emb)

        output = self.weight * audio_pred + (1 - self.weight) * video_pred
        return output


class SolutionToDifferentClass(nn.Module):
    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_pred = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.Linear(128, self.num_classes),
        )
        self.video_pred = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.Linear(128, self.num_classes),
        )
        self.weights = nn.Parameter(
            torch.tensor((0.7, 0.49, 0.45, 0.4, 0.5, 0.55, 0.5, 0.55, 0.55, 0.55)),
            requires_grad=True,
        )

    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)
        audio_pred = self.audio_pred(audio_emb)

        video_emb = video_feat.mean(1)
        video_pred = self.video_pred(video_emb)

        class_weights = self.weights
        output = audio_pred * class_weights + video_pred * (1 - class_weights)
        print(class_weights)
        return output


class MyModel(nn.Module):
    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc_q = nn.Linear(audio_emb_dim, 128)
        self.fc_k = nn.Linear(audio_emb_dim, 128)
        self.fc_v = nn.Linear(audio_emb_dim, 128)
        self.mh_att = nn.MultiheadAttention(
            128, dropout=0.2, num_heads=1, batch_first=True,
        )
        self.bn1 = nn.LayerNorm(128)
        self.row_wise_fc = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=0.2),
        )
        self.bn2 = nn.LayerNorm(128)
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
        )
        self.outputlayer = nn.Sequential(
            nn.Linear(256, 128), nn.Linear(128, self.num_classes),
        )

    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        Q, K, V = self.fc_q(audio_feat), self.fc_k(audio_feat), self.fc_v(audio_feat)
        audio_feat, _ = self.mh_att(Q, K, V)
        audio_feat = self.bn1(audio_feat)
        audio_feat = audio_feat + self.bn2(self.row_wise_fc(audio_feat))
        audio_emb = audio_feat.mean(1)

        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)

        embed = torch.cat((audio_emb, video_emb), 1)
        output = self.outputlayer(embed)
        return output
