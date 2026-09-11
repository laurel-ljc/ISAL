"""Metric XY encoding and proprioceptive queries over a local height map."""
import torch
from torch import nn
from rsl_rl.networks import MLP


class PositionEncoding2D(nn.Module):
    def __init__(self, channels, embedding_dim, map_shape, resolution):
        super().__init__()
        rows, cols = map_shape
        x = (torch.arange(cols, dtype=torch.float32) - (cols - 1) / 2) * resolution
        y = (torch.arange(rows, dtype=torch.float32) - (rows - 1) / 2) * resolution
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        self.register_buffer("coordinates", torch.stack([xx, yy]).unsqueeze(0))
        self.projection = nn.Conv2d(channels + 2, embedding_dim, 1)

    def forward(self, features):
        xy = self.coordinates.expand(features.shape[0], -1, -1, -1)
        return self.projection(torch.cat([features, xy], dim=1)).flatten(2).transpose(1, 2)


class TerrainAttention(nn.Module):
    def __init__(self, proprio_dim, map_shape=(11, 17), map_resolution=0.1,
                 embedding_dim=32, cnn_channels=(16, 32, 32), query_hidden_dims=(128,), num_heads=4):
        super().__init__()
        if len(map_shape) != 2 or min(map_shape) < 1 or map_resolution <= 0:
            raise ValueError("map_shape must contain positive (rows, columns), and resolution must be positive")
        if embedding_dim < 1 or num_heads < 1 or embedding_dim % num_heads:
            raise ValueError("embedding_dim must be positive and divisible by num_heads")
        if not cnn_channels or min(cnn_channels) < 1:
            raise ValueError("cnn_channels must be nonempty and positive")
        self.map_shape = tuple(map_shape)
        layers = []
        channels = 1
        for output_channels in cnn_channels:
            layers.extend([nn.Conv2d(channels, output_channels, 3, padding=1, padding_mode="replicate"), nn.ELU()])
            channels = output_channels
        self.policy_encoder = nn.Sequential(*layers)
        self.position_encoding = PositionEncoding2D(channels, embedding_dim, map_shape, map_resolution)
        self.query = MLP(proprio_dim, embedding_dim, list(query_hidden_dims), "elu")
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=0.0, batch_first=True)
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def scan_to_image(self, scan):
        if scan.ndim != 2 or scan.shape[1] != self.map_shape[0] * self.map_shape[1]:
            raise ValueError(f"Expected flat XY scan with {self.map_shape[0] * self.map_shape[1]} points, got {scan.shape}")
        return scan.reshape(scan.shape[0], 1, *self.map_shape)

    def encode_features(self, scan):
        return self.policy_encoder(self.scan_to_image(scan))

    def attend(self, proprioception, features):
        # Future affordance fusion belongs before this position encoding call.
        tokens = self.token_norm(self.position_encoding(features))
        query = self.query_norm(self.query(proprioception)).unsqueeze(1)
        output, _ = self.attention(query, tokens, tokens, need_weights=False)
        return self.output_norm(output.squeeze(1))

    def forward(self, proprioception, scan):
        return self.attend(proprioception, self.encode_features(scan))
