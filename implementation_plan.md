# Tensor Logic Implementation Plan

This plan outlines the implementation of a sample "Tensor Logic" system based on the paper "Tensor Logic: The Language of AI".
The goal is to create a working demonstration of the core concepts: representing relations as tensors and rules as tensor equations (einsums).

## User Review Required
> [!NOTE]
> I will use `numpy` for the backend tensor operations.
> The implementation will focus on **Inference** (Forward Chaining) as described in Section 3.2.
> I will implement the "Aunt" example and a simple "MLP" forward pass example from the paper.

## Proposed Changes

### Core Library
#### [NEW] [tensor_logic.py](file:///Users/johngraves/Documents/20251119/tensor_logic.py)
- `Tensor` class:
    - Wraps a `numpy.ndarray`.
    - Stores dimension names (indices).
- `Equation` class:
    - Represents `LHS = RHS`.
    - Parses string representation like `Aunt[x,z] = Sister[x,y] * Parent[y,z]`.
- `Program` class:
    - Manages a set of Tensors and Equations.
    - Implements `forward_chaining` to execute equations.

### Demos
#### [NEW] [demo_family.py](file:///Users/johngraves/Documents/20251119/demo_family.py)
- Implements the social network/family example from the paper.
- Defines `Sister` and `Parent` relations.
- Infers `Aunt`.

#### [NEW] [demo_mlp.py](file:///Users/johngraves/Documents/20251119/demo_mlp.py)
- Implements a single-layer perceptron.
- `Y[i] = sigmoid(W[i,j] * X[j])`.

## Verification Plan

### Automated Tests
I will create a test script `test_tensor_logic.py` to verify the correctness of the implementation.
- Test 1: Verify `Aunt` inference logic manually.
- Test 2: Verify MLP computation against a standard numpy implementation.

Command:
```bash
python3 test_tensor_logic.py
```

### Manual Verification
Run the demos and check output:
```bash
python3 demo_family.py
python3 demo_mlp.py
```
