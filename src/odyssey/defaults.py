from typing import Protocol, final

from torch import Tensor

from .compute.core import Compute
from .objective import Objective


class Result(Protocol):
    @property
    def loss(self) -> Tensor: ...


@final
class DefaultIteration[*ModelsTs, BatchT, ResultT: Result]:
    def training_step(
        self,
        objective: Objective[*ModelsTs, BatchT, ResultT],
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

    def inference_step(
        self,
        objective: Objective[*ModelsTs, BatchT, ResultT],
        compute: Compute[*ModelsTs],
        batch: BatchT,
    ) -> ResultT:
        with compute.autocast():
            result = objective.forward_pass(
                *compute.models, batch=batch, device=compute.device
            )

        return result
