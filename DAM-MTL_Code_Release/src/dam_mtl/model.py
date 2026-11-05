from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _conv_output_length(length: int, kernel: int, padding: int = 1, stride: int = 1) -> int:
    return (length + 2 * padding - kernel) // stride + 1


def flattened_length(input_length: int) -> int:
    length = input_length
    for kernel in (7, 3, 3):
        length = _conv_output_length(length, kernel)
        length //= 2
    return 128 * length


class ModulatedConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, descriptor_dim: int = 64):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="leaky_relu")
        init.zeros_(self.conv.bias)
        self.modulation = nn.Sequential(
            nn.Linear(descriptor_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 2 * out_channels),
            nn.Tanh(),
        )
        for layer in self.modulation:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.zeros_(layer.bias)

    def scale_shift(self, descriptor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        parameters = self.modulation(descriptor)
        channels = self.conv.out_channels
        return parameters[:, :channels].unsqueeze(-1), parameters[:, channels:].unsqueeze(-1)

    def forward(self, x: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        scale, shift = self.scale_shift(descriptor)
        return self.conv(x) * (scale + 1.0) + shift


class DAMMTLRegressor(nn.Module):
    def __init__(self, num_domains: int, input_length: int, descriptor_dim: int = 64):
        super().__init__()
        self.session_embedding = nn.Embedding(num_domains, descriptor_dim)
        self.time_projection = nn.Linear(1, descriptor_dim)
        self.conv1 = ModulatedConv1d(1, 32, 7, descriptor_dim)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = ModulatedConv1d(32, 64, 3, descriptor_dim)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = ModulatedConv1d(64, 128, 3, descriptor_dim)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_length(input_length), 256)
        self.fc2 = nn.Linear(256, 2)

    def session_descriptor(self, domain_ids: torch.Tensor) -> torch.Tensor:
        denominator = max(1, self.session_embedding.num_embeddings - 1)
        time_index = domain_ids.float().unsqueeze(-1) / denominator
        return self.session_embedding(domain_ids) + self.time_projection(time_index)

    def _encode(self, x: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.bn1(self.conv1(x, descriptor))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x, descriptor))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x, descriptor))))
        return self.flatten(x)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        descriptor = self.session_descriptor(domain_ids)
        features = self._encode(x, descriptor)
        features = self.dropout(features)
        return self.fc2(torch.relu(self.fc1(features)))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def functional_forward(
    model: DAMMTLRegressor,
    parameters: Dict[str, torch.Tensor],
    x: torch.Tensor,
    domain_ids: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    def linear(name: str, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, parameters[name + ".weight"], parameters[name + ".bias"])

    def batch_norm(name: str, value: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(
            value,
            model.get_buffer(name + ".running_mean"),
            model.get_buffer(name + ".running_var"),
            parameters[name + ".weight"],
            parameters[name + ".bias"],
            training=training,
            momentum=0.1,
        )

    def modulated_block(prefix: str, value: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        conv_weight = parameters[prefix + ".conv.weight"]
        conv_bias = parameters[prefix + ".conv.bias"]
        value = F.conv1d(value, conv_weight, conv_bias, padding=1)
        hidden = descriptor
        hidden = linear(prefix + ".modulation.0", hidden)
        hidden = F.layer_norm(hidden, (128,), parameters[prefix + ".modulation.1.weight"], parameters[prefix + ".modulation.1.bias"])
        hidden = F.gelu(hidden)
        hidden = linear(prefix + ".modulation.3", hidden)
        hidden = F.layer_norm(hidden, (256,), parameters[prefix + ".modulation.4.weight"], parameters[prefix + ".modulation.4.bias"])
        hidden = F.gelu(hidden)
        hidden = torch.tanh(linear(prefix + ".modulation.6", hidden))
        channels = conv_weight.shape[0]
        scale = hidden[:, :channels].unsqueeze(-1)
        shift = hidden[:, channels:].unsqueeze(-1)
        value = value * (scale + 1.0) + shift
        block_number = prefix[-1]
        value = batch_norm("bn" + block_number, value)
        return F.max_pool1d(torch.relu(value), 2)

    denominator = max(1, model.session_embedding.num_embeddings - 1)
    time_index = domain_ids.float().unsqueeze(-1) / denominator
    descriptor = F.embedding(domain_ids, parameters["session_embedding.weight"])
    descriptor = descriptor + F.linear(time_index, parameters["time_projection.weight"], parameters["time_projection.bias"])
    x = modulated_block("conv1", x, descriptor)
    x = modulated_block("conv2", x, descriptor)
    x = modulated_block("conv3", x, descriptor)
    x = x.flatten(1)
    x = F.dropout(x, 0.2, training=training)
    x = torch.relu(linear("fc1", x))
    return linear("fc2", x)
