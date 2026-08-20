from typing import Protocol, final, override

import torch
from torch import Tensor

from odyssey.iteration import Iteration

from .compute.core import Compute


class Result(Protocol):
    @property
    def loss(self) -> Tensor: ...


class DefaultObjective[*ModelsTs, BatchT, ResultT: Result](Protocol):
    def forward_pass(
        self, *models: *ModelsTs, batch: BatchT, device: torch.device
    ) -> ResultT: ...


@final
class DefaultIteration[*ModelsTs, BatchT, ResultT: Result](
    Iteration[*ModelsTs, DefaultObjective[*ModelsTs, BatchT, ResultT], BatchT, ResultT]
):
    @override
    def training_step(
        self,
        objective: DefaultObjective[*ModelsTs, BatchT, ResultT],
        compute: Compute[*ModelsTs],
        batch: BatchT,
        divisor: float,
    ) -> ResultT:
        with compute.autocast():
            result = objective.forward_pass(
                *compute.models, batch=batch, device=compute.device
            )
        compute.backward_pass(result.loss / divisor)
        return result

    @override
    def inference_step(
        self,
        objective: DefaultObjective[*ModelsTs, BatchT, ResultT],
        compute: Compute[*ModelsTs],
        batch: BatchT,
    ) -> ResultT:
        with compute.autocast():
            result = objective.forward_pass(
                *compute.models, batch=batch, device=compute.device
            )

        return result
