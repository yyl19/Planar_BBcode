# Planar Fault-Tolerant Quantum Computation with Low Overhead

This repository contains the code implementation accompanying the paper:

> **Planar Fault-Tolerant Quantum Computation with Low Overhead**  
> [arXiv:2506.18061](http://arxiv.org/abs/2506.18061)

The code focuses on generating planar quantum error-correcting codes and performing various logical Pauli measurements in a fault-tolerant manner using the proposed **planar BB code architecture**. An example implementation for the [[162, 8, 7]] code is provided.

## 📂 Code Structure

| File            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `planarBBcode.py` | Generates the planar BB code structure, including `Hx`, `Hz` stabilizer matrices. |
| `Xm.py`         | Performs **single-block** $X$ and $XX$ logical Pauli measurements.          |
| `Zm.py`         | Performs **single-block** $Z$ and $ZZ$ logical Pauli measurements.          |
| `jointx.py`     | Performs **two-block** $X$-type joint logical measurements.                 |
| `jointz.py`     | Performs **two-block** $Z$-type joint logical measurements.                 |

## 🧪 Example: [[162, 8, 7]] Code

We provide an example using the [[162, 8, 7]] planar BB code constructed in `planarBBcode.py`. This example demonstrates how to build the code and simulate logical Pauli measurements.

### 1. Generate the Code

```bash
python planarBBcode.py
