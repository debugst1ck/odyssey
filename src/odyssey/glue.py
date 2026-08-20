from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributed import ReduceOp

from .iteration import Phase


class StopTraining(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ComputeHandle[*ModelsTs]:
    models: tuple[*ModelsTs]
    device: torch.device
    is_main_process: bool
    reduce: Callable[[Tensor, ReduceOp.RedOpType], Tensor]
    state_dicts: Callable[[], Sequence[dict[str, Tensor]]]


@dataclass(frozen=True, slots=True)
class EpochTelemetry[*ModelsTs]:
    handle: ComputeHandle[*ModelsTs]
    is_training: bool
    epoch_index: int
    total_batches: int


@dataclass(frozen=True, slots=True)
class BatchTelemetry[*ModelsTs, ObjectiveT, BatchT, ResultT](EpochTelemetry[*ModelsTs]):
    batch_index: int
    is_accumulation_boundary: bool
    phase: Phase[*ModelsTs, ObjectiveT, BatchT, ResultT]


@dataclass(frozen=True, slots=True)
class StepTelemetry[*ModelsTs](EpochTelemetry[*ModelsTs]):
    optimizer_step: int
