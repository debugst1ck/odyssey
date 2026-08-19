from collections.abc import Sequence
from typing import Protocol

from torch.optim import Optimizer

from .compute.core import Compute
from .objective import Objective


class Iteration[*ModelsTs, BatchT, ResultT](Protocol):
    def training_step(
        self,
        objective: Objective[*ModelsTs, BatchT, ResultT],
        compute: Compute[*ModelsTs],
        batch: BatchT,
        divisor: float,
    ) -> ResultT: ...

    def inference_step(
        self,
        objective: Objective[*ModelsTs, BatchT, ResultT],
        compute: Compute[*ModelsTs],
        batch: BatchT,
    ) -> ResultT: ...


class Phase[*ModelsTs, BatchT, ResultT](Protocol):
    iteration: Iteration[*ModelsTs, BatchT, ResultT]
    optimizers: Sequence[Optimizer]
