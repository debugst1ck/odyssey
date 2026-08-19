from .glue import BatchTelemetry, EpochTelemetry, StepTelemetry


class Plugin[*ModelsTs, BatchT, ResultT]:
    def on_epoch_begin(self, _telemetry: EpochTelemetry[*ModelsTs]) -> None:
        pass

    def on_batch_begin(
        self, _telemetry: BatchTelemetry[*ModelsTs, BatchT, ResultT]
    ) -> None:
        pass

    def on_batch_end(
        self, _telemetry: BatchTelemetry[*ModelsTs, BatchT, ResultT], _result: ResultT
    ) -> None:
        pass

    def on_optimizer_step(self, _telemetry: StepTelemetry[*ModelsTs]) -> None:
        pass

    def on_epoch_end(self, _telemetry: EpochTelemetry[*ModelsTs]) -> None:
        pass
