# Portable Synchronization Primitives for MLIR

`mlir-sync` is intended to provide synchronization primitives that are portable across platforms and suitable for MLIR-related runtime or systems work.

## Goals

1. **Inlined fast paths**

   Optimize the uncontended case with lightweight, inlined operations to minimize synchronization overhead in performance-sensitive paths.

2. **Portable futex-based synchronization for multiple platforms**

   Provide a futex-style synchronization layer that can be adapted to different operating systems while preserving a consistent interface and efficient blocking behavior.

## License

Licensed under either of the following, at your option:

- Apache License, Version 2.0
- MIT license
