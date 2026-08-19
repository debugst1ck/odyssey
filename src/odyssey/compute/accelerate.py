from collections.abc import Sequence
from contextlib import nullcontext
from typing import cast, final, override

import torch
from torch import Tensor, nn
from torch.distributed import ReduceOp
from torch.optim import Optimizer

from .core import Compute


@final
class AcceleratedCompute[*ModelsTs](Compute[*ModelsTs]):
    def __init__(
        self,
        models: tuple[*ModelsTs],
        mixed_precision: bool = True,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if torch.accelerator.is_available():
            self._device = cast(torch.device, torch.accelerator.current_accelerator())
        else:
            raise RuntimeError("Accelerator is not available.")

        self._models: tuple[*ModelsTs] = models

        for model in self._models:
            assert isinstance(model, nn.Module), (
                "All models must be instances of torch.nn.Module."
            )
            _ = model.to(self._device)

        self.mixed_precision = mixed_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # GradScaler is only needed for float16 (bfloat16 doesn't need scaling)
        use_scaler = mixed_precision and (mixed_precision_dtype == torch.float16)
        self.scaler = torch.amp.GradScaler(enabled=use_scaler)

    @property
    @override
    def models(self) -> tuple[*ModelsTs]:
        return self._models

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @property
    @override
    def is_main_process(self) -> bool:
        return True  # Single process is always main

    @override
    def autocast(self) -> torch.amp.autocast:
        return torch.amp.autocast(
            device_type=self._device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision,
        )

    @override
    def no_sync(self) -> nullcontext[None]:
        return nullcontext()  # Always sync

    @override
    def zero_gradients(self, optimizers: Sequence[Optimizer]) -> None:
        for optimizer in optimizers:
            optimizer.zero_grad()

    @override
    def backward_pass(self, loss: Tensor) -> None:
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    @override
    def step_optimizers(self, optimizers: Sequence[Optimizer], clipping: float) -> bool:
        if self.scaler.is_enabled():
            for optimizer in optimizers:
                self.scaler.unscale_(optimizer)
        if clipping > 0.0:
            for model in self._models:
                _ = torch.nn.utils.clip_grad_norm_(
                    cast(nn.Module, model).parameters(), clipping
                )
        initial_scale = self.scaler.get_scale()
        for optimizer in optimizers:
            if self.scaler.is_enabled():
                _ = self.scaler.step(optimizer)
            else:
                optimizer.step()
        if self.scaler.is_enabled():
            self.scaler.update()
        # Return True if the step was successful, False if skipped due to inf/nan
        return self.scaler.get_scale() >= initial_scale

    @override
    def synchronize(self) -> None:
        torch.accelerator.synchronize(self._device)

    @override
    def train(self, enabled: bool = True) -> None:
        for model in self._models:
            _ = cast(nn.Module, model).train(enabled)

    @override
    def reduce(self, tensor: Tensor, op: ReduceOp.RedOpType = ReduceOp.SUM) -> Tensor:
        return tensor  # No-op for single-process, single-device compute.
