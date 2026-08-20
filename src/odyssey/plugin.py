from .glue import BatchTelemetry, EpochTelemetry, StepTelemetry


class Plugin[*ModelsTs, ObjectiveT, BatchT, ResultT]:
    def on_epoch_begin(
        self, _telemetry: EpochTelemetry[*ModelsTs, ObjectiveT, BatchT, ResultT]
    ) -> None:
        pass

    def on_batch_begin(
        self, _telemetry: BatchTelemetry[*ModelsTs, ObjectiveT, BatchT, ResultT]
    ) -> None:
        pass

    def on_batch_end(
        self,
        _telemetry: BatchTelemetry[*ModelsTs, ObjectiveT, BatchT, ResultT],
        _result: ResultT,
    ) -> None:
        pass

    def on_optimizer_step(
        self, _telemetry: StepTelemetry[*ModelsTs, ObjectiveT, BatchT, ResultT]
    ) -> None:
        pass

    def on_epoch_end(
        self, _telemetry: EpochTelemetry[*ModelsTs, ObjectiveT, BatchT, ResultT]
    ) -> None:
        pass
