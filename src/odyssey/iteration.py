from collections.abc import Sequence
from typing import Protocol, final

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


@final
class Phase[*ModelsTs, ObjectiveT, BatchT, ResultT]:
    def __init__(
        self,
        iteration: Iteration[*ModelsTs, ObjectiveT, BatchT, ResultT],
        optimizers: Sequence[Optimizer],
    ) -> None:
        self.iteration = iteration
        self.optimizers = optimizers
