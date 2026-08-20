from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from torch.optim import Optimizer

from .compute.core import Compute


class Iteration[*ModelsTs, ObjectiveT, BatchT, ResultT](Protocol):
    def training_step(
        self,
        objective: ObjectiveT,
        compute: Compute[*ModelsTs],
        batch: BatchT,
        divisor: float,
    ) -> ResultT: ...

    def inference_step(
        self,
        objective: ObjectiveT,
        compute: Compute[*ModelsTs],
        batch: BatchT,
    ) -> ResultT: ...


@dataclass(frozen=True, slots=True)
class Phase[*ModelsTs, ObjectiveT, BatchT, ResultT]:
    iteration: Iteration[*ModelsTs, ObjectiveT, BatchT, ResultT]
    optimizers: Sequence[Optimizer]
