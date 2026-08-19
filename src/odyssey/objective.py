from typing import Protocol

import torch


class Objective[*ModelsTs, BatchT, ResultT](Protocol):
    def forward_pass(
        self, *models: *ModelsTs, batch: BatchT, device: torch.device
    ) -> ResultT: ...
