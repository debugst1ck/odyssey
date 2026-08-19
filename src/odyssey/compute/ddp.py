import os
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from typing import cast, final, override

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from .core import Compute


@final
class DistributedDataParallelCompute[*ModelsTs](Compute[*ModelsTs]):
    """Multi-GPU compute backend using torch.nn.parallel.DistributedDataParallel."""

    def __init__(
        self,
        models: tuple[*ModelsTs],
        mixed_precision: bool = True,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before instantiating DDP Compute."
            )

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self._global_rank = dist.get_rank()

        if torch.accelerator.is_available():
            torch.accelerator.set_device_index(local_rank)
            acc = torch.accelerator.current_accelerator()

            if acc is not None:
                self._device = torch.device(f"{acc.type}:{local_rank}")
            else:
                raise RuntimeError(
                    "Accelerator is not available, but torch.distributed is initialized."
                )
        else:
            raise RuntimeError(
                "Accelerator is not available, but torch.distributed is initialized."
            )

        self._models: tuple[*ModelsTs] = models

        for model in self._models:
            assert isinstance(model, nn.Module), (
                "All models must be instances of torch.nn.Module."
            )
            _ = model.to(self._device)

        backend = dist.get_backend()
        device_ids = [local_rank] if backend != "gloo" else None

        self._ddp_models = tuple(
            DDP(
                model,
                device_ids=device_ids,
                output_device=local_rank if backend != "gloo" else None,
            )
            for model in self._models
        )

        self.mixed_precision = mixed_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        use_scaler = mixed_precision and (mixed_precision_dtype == torch.float16)
        self.scaler = torch.amp.GradScaler(enabled=use_scaler)

    @property
    @override
    def models(self) -> tuple[*ModelsTs]:
        # THE LIE: We tell the type system these are the pure models.
        # At runtime, they are DDP wrappers, but DDP forwards __call__
        # and custom methods to the pure model perfectly.
        return cast(tuple[*ModelsTs], self._ddp_models)

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @property
    @override
    def is_main_process(self) -> bool:
        return self._global_rank == 0

    @override
    def autocast(self) -> torch.amp.autocast:
        return torch.autocast(
            device_type=self._device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision,
        )

    @override
    @contextmanager
    def no_sync(self):
        # Use ExitStack to pause sync for all models simultaneously
        with ExitStack() as stack:
            for model in self._ddp_models:
                stack.enter_context(model.no_sync())
            yield

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
            for model in self._ddp_models:
                _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)

        initial_scale = self.scaler.get_scale()
        for optimizer in optimizers:
            if self.scaler.is_enabled():
                _ = self.scaler.step(optimizer)
            else:
                optimizer.step()

        if self.scaler.is_enabled():
            self.scaler.update()

        return self.scaler.get_scale() >= initial_scale

    @override
    def synchronize(self) -> None:
        if dist.is_initialized():
            dist.barrier()
        torch.accelerator.synchronize(self._device)

    @override
    def train(self, enabled: bool = True) -> None:
        for model in self._ddp_models:
            _ = model.train(enabled)

    @override
    def reduce(self, tensor: Tensor, op: ReduceOp.RedOpType = ReduceOp.SUM) -> Tensor:
        cloned = tensor.clone()
        _ = dist.all_reduce(cloned, op=op)
        return cloned

    @override
    def state_dicts(self) -> Sequence[dict[str, Tensor]]:
        return [
            {k: v.cpu() for k, v in cast(nn.Module, m).state_dict().items()}
            for m in self._models
        ]
