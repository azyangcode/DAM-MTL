from __future__ import annotations

import copy
import logging
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import RegressionConfig
from .data import SessionData, source_indices, split_source, split_target, support_sample_count
from .metrics import regression_metrics
from .model import DAMMTLRegressor, count_parameters, functional_forward


LOGGER = logging.getLogger(__name__)


class TensorDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, domain_ids: np.ndarray = None):
        self.x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(1)
        self.y = torch.from_numpy(np.ascontiguousarray(y))
        self.domain_ids = None if domain_ids is None else torch.from_numpy(np.ascontiguousarray(domain_ids))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        if self.domain_ids is None:
            return self.x[index], self.y[index]
        return self.x[index], self.y[index], self.domain_ids[index]


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    drop_last = len(dataset) >= batch_size and len(dataset) % batch_size == 1
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


@dataclass
class FoldResult:
    monkey: str
    protocol: str
    fold: int
    session: str
    seed: int
    cc_x: float
    cc_y: float
    cc_avg: float
    rmse_x: float
    rmse_y: float
    rmse_avg: float
    r2_x: float
    r2_y: float
    r2_avg: float
    mae_x: float
    mae_y: float
    mae_avg: float
    n_support: int
    n_validation: int
    n_query: int
    elapsed_seconds: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DAMMTLTrainer:
    def __init__(self, input_length: int, config: RegressionConfig, device: torch.device, seed: int):
        self.input_length = input_length
        self.config = config
        self.device = device
        self.seed = seed
        self.model: DAMMTLRegressor = None

    def build_model(self, number_of_domains: int) -> DAMMTLRegressor:
        set_random_seed(self.seed)
        self.model = DAMMTLRegressor(number_of_domains, self.input_length).to(self.device)
        LOGGER.info("Model parameters: %d", count_parameters(self.model))
        return self.model

    def train_stage1(self, sources: Sequence[Tuple[np.ndarray, np.ndarray]]) -> None:
        set_random_seed(self.seed)
        xs, ys, domain_ids = [], [], []
        for domain_id, (x, y) in enumerate(sources):
            xs.append(x)
            ys.append(y)
            domain_ids.append(np.full(len(y), domain_id, dtype=np.int64))
        dataset = TensorDataset(np.concatenate(xs), np.concatenate(ys), np.concatenate(domain_ids))
        loader = make_loader(dataset, self.config.stage1.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.stage1.learning_rate,
            momentum=self.config.stage1.momentum,
        )
        criterion = nn.SmoothL1Loss()
        self.model.train()
        for epoch in range(1, self.config.stage1.epochs + 1):
            total = 0.0
            for x, y, ids in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                ids = ids.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(x, ids), y)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(x)
            LOGGER.info("stage1 epoch=%d loss=%.6f", epoch, total / len(dataset))

    def train_stage2(self, sources: Sequence[Tuple[np.ndarray, np.ndarray]]) -> None:
        set_random_seed(self.seed)
        tasks = []
        for x, y in sources:
            x_train, y_train, x_query, y_query = split_source(x, y, self.config.protocol.source_meta_fraction)
            tasks.append(
                (
                    make_loader(TensorDataset(x_train, y_train), self.config.stage2.batch_size, shuffle=True),
                    make_loader(TensorDataset(x_query, y_query), self.config.stage2.batch_size, shuffle=True),
                )
            )

        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.session_embedding.parameters(), "lr": self.config.stage2.outer_learning_rate},
                {"params": [p for n, p in self.model.named_parameters() if "modulation" in n], "lr": self.config.stage2.outer_learning_rate},
                {"params": self.model.fc1.parameters(), "lr": self.config.stage2.outer_learning_rate},
                {"params": self.model.fc2.parameters(), "lr": self.config.stage2.outer_learning_rate},
            ],
            weight_decay=self.config.stage2.weight_decay,
        )
        modulation_names = {name for name, _ in self.model.named_parameters() if "modulation" in name}
        self.model.train()

        for iteration in range(1, self.config.stage2.iterations + 1):
            anchors = {}
            if self.config.stage2.anchor_coefficient > 0:
                anchors = {
                    name: parameter.detach().clone()
                    for name, parameter in self.model.named_parameters()
                    if name in modulation_names
                }

            task_order = list(range(len(tasks)))
            np.random.shuffle(task_order)
            running_loss = 0.0
            for task_id in task_order:
                support_loader, query_loader = tasks[task_id]
                fast_base = OrderedDict(
                    (
                        name,
                        parameter.detach().clone(memory_format=torch.preserve_format).requires_grad_(True),
                    )
                    for name, parameter in self.model.named_parameters()
                )
                fast_weights = fast_base
                for _ in range(self.config.stage2.inner_steps):
                    x, y = next(iter(support_loader))
                    x = x.to(self.device)
                    y = y.to(self.device)
                    ids = torch.full((len(x),), task_id, dtype=torch.long, device=self.device)
                    prediction = functional_forward(self.model, fast_weights, x, ids, training=True)
                    loss = F.smooth_l1_loss(prediction, y)
                    gradients = torch.autograd.grad(
                        loss,
                        list(fast_weights.values()),
                        create_graph=True,
                        allow_unused=True,
                    )
                    fast_weights = OrderedDict(
                        (
                            name,
                            parameter - self.config.stage2.inner_learning_rate * gradient
                            if gradient is not None
                            else parameter,
                        )
                        for (name, parameter), gradient in zip(fast_weights.items(), gradients)
                    )

                x, y = next(iter(query_loader))
                x = x.to(self.device)
                y = y.to(self.device)
                ids = torch.full((len(x),), task_id, dtype=torch.long, device=self.device)
                prediction = functional_forward(self.model, fast_weights, x, ids, training=False)
                meta_loss = F.smooth_l1_loss(prediction, y)
                outer_gradients = torch.autograd.grad(
                    meta_loss,
                    list(fast_base.values()),
                    allow_unused=True,
                )
                optimizer.zero_grad(set_to_none=True)
                named_parameters = dict(self.model.named_parameters())
                for (name, _), gradient in zip(fast_base.items(), outer_gradients):
                    if gradient is not None:
                        parameter = named_parameters[name]
                        if parameter.grad is None:
                            parameter.grad = gradient.detach().clone()
                        else:
                            parameter.grad.add_(gradient.detach())
                if anchors:
                    for name, parameter in self.model.named_parameters():
                        if name in anchors and parameter.grad is not None:
                            parameter.grad.add_(
                                2.0 * self.config.stage2.anchor_coefficient * (parameter.detach() - anchors[name])
                            )
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.stage2.gradient_clip_norm)
                optimizer.step()
                running_loss += float(meta_loss.item())
            LOGGER.info("stage2 iteration=%d loss=%.6f", iteration, running_loss / len(tasks))

    def adapt_target(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
        x_query: np.ndarray,
        y_query: np.ndarray,
        target_domain_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        set_random_seed(self.seed)
        train_loader = make_loader(
            TensorDataset(x_train, y_train),
            self.config.stage3.batch_size,
            shuffle=True,
        )
        validation_loader = make_loader(
            TensorDataset(x_validation, y_validation),
            self.config.stage3.batch_size,
            shuffle=False,
        )
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.stage3.learning_rate,
            momentum=self.config.stage3.momentum,
            nesterov=True,
        )
        criterion = nn.SmoothL1Loss()
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        self.model.train()

        for epoch in range(1, self.config.stage3.max_epochs + 1):
            epoch_loss = 0.0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                ids = torch.full((len(x),), target_domain_id, dtype=torch.long, device=self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(x, ids), y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(x)

            self.model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for x, y in validation_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    ids = torch.full((len(x),), target_domain_id, dtype=torch.long, device=self.device)
                    validation_loss += float(criterion(self.model(x, ids), y).item()) * len(x)
            validation_loss /= max(1, len(y_validation))
            LOGGER.info(
                "stage3 epoch=%d train_loss=%.6f validation_loss=%.6f",
                epoch,
                epoch_loss / len(y_train),
                validation_loss,
            )
            self.model.train()

            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = copy.deepcopy({name: value.detach().cpu() for name, value in self.model.state_dict().items()})
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.stage3.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict({name: value.to(self.device) for name, value in best_state.items()})
        return self.predict(x_query, y_query, target_domain_id, self.config.evaluation.query_batch_size)

    def predict(
        self,
        x_query: np.ndarray,
        y_query: np.ndarray,
        target_domain_id: int,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        predictions = []
        targets = []
        loader = make_loader(TensorDataset(x_query, y_query), batch_size, shuffle=False)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                ids = torch.full((len(x),), target_domain_id, dtype=torch.long, device=self.device)
                predictions.append(self.model(x, ids).cpu().numpy())
                targets.append(y.numpy())
        return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def train_and_evaluate(
    sessions: Sequence[SessionData],
    config: RegressionConfig,
    protocol: str,
    fold: int,
    seed_index: int,
    device: torch.device,
) -> FoldResult:
    sources = source_indices(fold, len(sessions), protocol)
    target = sessions[fold]
    support_samples = support_sample_count(
        config.protocol.calibration_seconds,
        config.data.window_ms,
        config.data.step_ms,
    )
    x_train, y_train, x_validation, y_validation, x_query, y_query = split_target(
        target.x,
        target.y,
        support_samples,
        config.protocol.validation_fraction,
    )
    seed = config.seeds[seed_index] + fold
    trainer = DAMMTLTrainer(config.data.channels, config, device, seed)
    trainer.build_model(len(sources) + 1)
    started = time.time()
    source_data = [(sessions[index].x, sessions[index].y) for index in sources]
    trainer.train_stage1(source_data)
    trainer.train_stage2(source_data)
    prediction, observed = trainer.adapt_target(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_query,
        y_query,
        target_domain_id=len(sources),
    )
    metrics = regression_metrics(prediction, observed)
    return FoldResult(
        monkey=target.monkey,
        protocol=protocol,
        fold=fold,
        session=target.filename,
        seed=seed,
        n_support=len(x_train),
        n_validation=len(x_validation),
        n_query=len(x_query),
        elapsed_seconds=time.time() - started,
        **metrics,
    )
