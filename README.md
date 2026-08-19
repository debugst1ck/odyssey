# Odyssey

Odyssey is a training library for PyTorch modules. It provides a set of tools and utilities to facilitate the training and evaluation of deep learning models, making it easier for researchers and practitioners to experiment with different architectures and training strategies.

## Why not PyTorch Lightning?

While PyTorch Lightning is a popular choice for simplifying the training loop and providing a high-level interface for PyTorch, Odyssey aims to offer a more flexible and lightweight alternative.

## Design Principles

Odyssey operates on the principle of Inversion of Control (IoC), allowing users to have more control over the training process while still benefiting from a structured framework. It is neatly partitioned to 5 different modules, each with a specific purpose and responsibility:

- Objectives: This module contains strictly the forward functions of the model, the mathematical operations that define the model's behavior.
- Iteration: This module contains the training loop, for forward and backward passes.
- Plugins: This module contains schedulers, progress bars, checkpointing, and many other utilities that can be used to enhance the training process.
- Compute: This module defines the compute strategy, whether it be single device, distributed data parallel (DDP), or fully sharded data parallel (FSDP2).
- Orchestration: This module is responsible for orchestrating the training process, coordinating the objectives, stepping optimizers, and managing the overall flow of the training loop.

## Warning to authors, contributers and users

Odyssey is designed to support multiple modules, optimizers and schedulers, but due to how distibuted data parallel works as a wrapper, it assumes we only use the forward function of the module, at the top level, this means you cannot call submodules when calculating the loss. It is because we pass the DDP module rather than the actual nn.Module, and the DDP module only has the forward function.

But if you're using the single device compute or FSDP2, you can use submodules without restrictions.
