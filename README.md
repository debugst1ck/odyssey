# Odyssey

Odyssey is a modern, lightweight, type-safe training orchestrator for PyTorch, built from the ground up using advanced Python type hinting (PEP 646 Type Parameter Syntax / Type Var Tuples). It provides a clean separation of concerns between compute hardware management, objectives, training phases, and telemetry/plugins, making complex setups like multi-model loops (e.g., GANs) and gradient accumulation seamless.

## Features

- Type-Safe Generic Architecture: Leverages Python's modern typing capabilities (`*ModelsTs`) to ensure type safety across models, batches, and objectives.
- Flexible Gradient Accumulation: Supports two accumulation modes:
- `"stream"`: Processes batches sequentially to conserve CPU/RAM.
- `"block"`: Pre-loads chunks into memory for better mathematical accuracy in multi-model series training (e.g., GANs).

- Decoupled Plugins & Telemetry: Easily hook into lifecycle events (`on_epoch_begin`, `on_batch_begin`, `on_batch_end`, `on_optimizer_step`, `on_epoch_end`) for logging, checkpointing, or metrics tracking.
- Phase & Iteration Abstraction: Distinctly separates optimization phases, training steps, and forward-pass logic from the core orchestrator loop.
- Distributed & Mixed Precision Ready: Built-in hooks for gradient clipping, synchronization contexts, and distributed computing bounds.

## Core Components

1. `Orchestrator`: The central loop runner that handles epochs, batch splitting, mode switching (training/inference), and plugin callbacks.
2. `Compute`: Protocol defining hardware interactions (device placement, backward passes, gradient zeroing, optimizer steps, and distributed reductions).
3. `Objective`: Defines the core forward pass logic for your models given a data batch.
4. `Phase`: Combines an `Iteration` strategy with a specific sequence of `Optimizer`s.
5. `Plugin`: Lifecycle hooks for extending training behavior without cluttering the main loop.
