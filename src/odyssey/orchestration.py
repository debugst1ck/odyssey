import itertools
import math
from collections.abc import Iterable, Sequence, Sized
from contextlib import nullcontext
from typing import Literal, Protocol, final

from torch import inference_mode

from .compute.core import Compute
from .glue import (
    BatchTelemetry,
    ComputeHandle,
    EpochTelemetry,
    StepTelemetry,
    StopTraining,
)
from .iteration import Phase
from .objective import Objective
from .plugin import Plugin


class BoundedIterable[T](Iterable[T], Sized, Protocol):
    pass


@final
class Orchestrator[*ModelsTs, BatchT, ResultT]:
    def __init__(
        self,
        compute: Compute[*ModelsTs],
        objective: Objective[*ModelsTs, BatchT, ResultT],
        phases: Sequence[Phase[*ModelsTs, BatchT, ResultT]],
        *,
        plugins: Sequence[Plugin[*ModelsTs, BatchT, ResultT]] = (),
        accumulation_steps: int = 1,
        clipping: float = 1.0,
        accumulation_mode: Literal["stream", "block"] = "stream",
    ) -> None:
        self.compute = compute
        self.objective = objective
        self.phases = phases
        self.plugins = tuple(plugins)
        self.accumulation_steps = max(1, accumulation_steps)
        self.clipping = clipping
        self.accumulation_mode = accumulation_mode
        self.handle = ComputeHandle(
            models=self.compute.models,
            device=self.compute.device,
            is_main_process=self.compute.is_main_process,
            reduce=self.compute.reduce,
        )

        self._optimizer_step = 0
        self._epoch_index = 0

    def _chunk(
        self,
        dataloader: BoundedIterable[BatchT],
        epoch_telemetry: EpochTelemetry[*ModelsTs],
        is_training: bool,
    ) -> None:
        """
        Consumes more CPU memory, because it pre-loads the entire chunk into memory.
        But this is more mathematically accurate for multi-model setups like GANs, where models are trained in series.
        """
        iterator = iter(dataloader)
        chunk_count = math.ceil(epoch_telemetry.total_batches / self.accumulation_steps)
        for chunk_index in range(chunk_count):
            chunk = tuple(itertools.islice(iterator, self.accumulation_steps))
            for phase in self.phases:
                for local_index, batch in enumerate(chunk):
                    is_last_batch = local_index == (len(chunk) - 1)
                    global_index = chunk_index * self.accumulation_steps + local_index

                    batch_telemetry = BatchTelemetry(
                        epoch_telemetry.handle,
                        epoch_telemetry.is_training,
                        epoch_telemetry.epoch_index,
                        epoch_telemetry.total_batches,
                        global_index,
                        is_last_batch,
                        phase,
                    )

                    for plugin in self.plugins:
                        plugin.on_batch_begin(batch_telemetry)

                    if is_training:
                        sync_context = (
                            nullcontext() if is_last_batch else self.compute.no_sync()
                        )
                        with sync_context:
                            result = phase.iteration.training_step(
                                self.objective,
                                self.compute,
                                batch,
                                divisor=len(chunk),
                            )
                    else:
                        result = phase.iteration.inference_step(
                            self.objective, self.compute, batch
                        )

                    for plugin in self.plugins:
                        plugin.on_batch_end(batch_telemetry, result)
                if is_training:
                    optimized = self.compute.step_optimizers(
                        phase.optimizers, self.clipping
                    )
                    self.compute.zero_gradients(phase.optimizers)
                    if optimized:
                        self._optimizer_step += 1
                        step_telemetry = StepTelemetry(
                            epoch_telemetry.handle,
                            epoch_telemetry.is_training,
                            epoch_telemetry.epoch_index,
                            epoch_telemetry.total_batches,
                            self._optimizer_step,
                        )
                        for plugin in self.plugins:
                            plugin.on_optimizer_step(step_telemetry)

    def _stream(
        self,
        dataloader: BoundedIterable[BatchT],
        epoch_telemetry: EpochTelemetry[*ModelsTs],
        is_training: bool,
    ) -> None:
        """
        Consumes less CPU memory, because it processes batches one at a time.
        But this is less mathematically accurate for multi-model setups like GANs, where models are trained in series.
        """
        total_batches = epoch_telemetry.total_batches
        for batch_index, batch in enumerate(dataloader):
            is_last_batch = batch_index == (total_batches - 1)
            is_normal_boundary = ((batch_index + 1) % self.accumulation_steps) == 0
            is_sync_boundary = is_last_batch or is_normal_boundary

            for phase in self.phases:
                batch_telemetry = BatchTelemetry(
                    self.handle,
                    is_training,
                    self._epoch_index,
                    total_batches,
                    batch_index,
                    is_sync_boundary,
                    phase,
                )

                for plugin in self.plugins:
                    plugin.on_batch_begin(batch_telemetry)

                if is_training:
                    sync_ctx = (
                        nullcontext() if is_sync_boundary else self.compute.no_sync()
                    )
                    with sync_ctx:
                        result = phase.iteration.training_step(
                            self.objective,
                            self.compute,
                            batch,
                            float(self.accumulation_steps),
                        )
                else:
                    result = phase.iteration.inference_step(
                        self.objective, self.compute, batch
                    )

                for plugin in self.plugins:
                    plugin.on_batch_end(batch_telemetry, result)

            if is_training and is_sync_boundary:
                for phase in self.phases:
                    optimized = self.compute.step_optimizers(
                        phase.optimizers, self.clipping
                    )
                    self.compute.zero_gradients(phase.optimizers)
                    if optimized:
                        self._optimizer_step += 1
                        step_telemetry = StepTelemetry(
                            self.handle,
                            is_training,
                            self._epoch_index,
                            total_batches,
                            self._optimizer_step,
                        )
                        for plugin in self.plugins:
                            plugin.on_optimizer_step(step_telemetry)

    def run(
        self,
        dataloader: BoundedIterable[BatchT],
        is_training: bool = True,
    ) -> None:
        self.compute.train(is_training)

        epoch_telemetry = EpochTelemetry(
            self.handle, is_training, self._epoch_index, len(dataloader)
        )
        for plugin in self.plugins:
            plugin.on_epoch_begin(epoch_telemetry)
        context = nullcontext() if is_training else inference_mode()
        try:
            with context:
                match self.accumulation_mode:
                    case "stream":
                        self._stream(dataloader, epoch_telemetry, is_training)
                    case "block":
                        self._chunk(dataloader, epoch_telemetry, is_training)
                    case _:
                        raise ValueError(
                            f"Invalid accumulation mode: {self.accumulation_mode}"
                        )
        except StopTraining:
            pass
        finally:
            self.compute.synchronize()
            for plugin in self.plugins:
                plugin.on_epoch_end(epoch_telemetry)
