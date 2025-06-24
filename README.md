# Planar Fault-Tolerant Quantum Computation with Low Overhead

This repository contains the code implementation accompanying the paper:

> **Planar Fault-Tolerant Quantum Computation with Low Overhead**  
> [arXiv:2506.18061](http://arxiv.org/abs/2506.18061)

The code focuses on generating planar quantum error-correcting codes and performing various logical Pauli measurements in a fault-tolerant manner using the proposed **planar BB code craft**. An example implementation for the [[162, 8, 7]] code is provided.

## 📂 Code Structure

| File            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `planarBBcode.py` | Generates the planar BB code structure, including `Hx`, `Hz` stabilizer matrices. |
| `Xm.py`         | Performs **single-block** $X$ and $XX$ logical Pauli measurements.          |
| `Zm.py`         | Performs **single-block** $Z$ and $ZZ$ logical Pauli measurements.          |
| `jointx.py`     | Performs **two-block** $X$-type joint logical measurements.                 |
| `jointz.py`     | Performs **two-block** $Z$-type joint logical measurements.                 |

## 🧪 Example: [[162, 8, 7]] Code



### 1. Generate the Code

```bash
$ python planarBBcode.py
The [[162,8,7]] code has been saved to './data/hxhz_7_7.pkl'.
### 2. Performs **single-block** $X$ and $XX$
$ python Xm.py
========= Summary X  =========
X0
wid	n	dx	Before opt dz	After opt dz 
9	190	7	2	3
10	208	7	2	4
11	226	7	3	5
12	244	7	6	7

========= Summary XX =========
X0 X1
wid	n	dx	Before opt dz	After opt dz 
9	190	8	2	3
10	208	8	2	4
11	226	8	2	4
12	244	8	3	6
13	262	8	2	7
X0 X2
wid	n	dx	Before opt dz	After opt dz 
9	190	8	2	4
10	208	8	2	4
11	226	8	2	6
12	244	8	2	6
13	262	8	3	6
14	280	8	2	6
15	298	8	3	7
X0 X3
wid	n	dx	Before opt dz	After opt dz 
9	190	7	5	5
10	208	7	5	5
11	226	7	7	7
X0 X4
wid	n	dx	Before opt dz	After opt dz 
9	190	8	3	4
10	208	8	2	4
11	226	8	2	6
12	244	8	3	6
13	262	8	4	7
X0 X5
wid	n	dx	Before opt dz	After opt dz 
9	190	7	3	3
10	208	7	4	5
11	226	7	3	5
12	244	7	3	7
X0 X6
wid	n	dx	Before opt dz	After opt dz 
9	190	8	2	4
10	208	8	2	4
11	226	8	2	6
12	244	8	2	6
13	262	8	2	6
14	280	8	2	7
X0 X7
wid	n	dx	Before opt dz	After opt dz 
9	190	7	2	4
10	208	7	3	4
11	226	7	3	6
12	244	7	5	6
13	262	7	5	6
14	280	7	7	7
### 3. Performs **single-block** $Z$ and $ZZ$
$ python Zm.py
========= Summary Z =========
Z0
leng	n	dz	优化前dx	优化后dx 
9	192	9	2	4
10	210	9	2	5
11	228	9	2	6
12	246	9	3	6
13	264	9	4	7

========= Summary ZZ =========
Z0 Z1
leng	n	dz	优化前dx	优化后dx 
9	192	8	2	4
10	210	8	5	5
11	228	8	4	6
12	246	8	2	6
13	264	8	7	7
Z0 Z2
leng	n	dz	优化前dx	优化后dx 
9	192	8	2	5
10	210	8	2	6
11	228	8	2	6
12	246	8	2	6
13	264	8	2	6
14	282	8	3	7
Z0 Z3
leng	n	dz	优化前dx	优化后dx 
9	192	8	2	4
10	210	8	2	5
11	228	8	2	6
12	246	8	3	7
Z0 Z4
leng	n	dz	优化前dx	优化后dx 
9	192	9	2	3
10	210	9	2	4
11	228	9	2	5
12	246	9	2	6
13	264	9	3	6
14	282	9	4	6
15	300	9	4	7
Z0 Z5
leng	n	dz	优化前dx	优化后dx 
9	192	8	3	4
10	210	8	2	5
11	228	8	2	6
12	246	8	3	7
Z0 Z6
leng	n	dz	优化前dx	优化后dx 
9	192	8	4	4
10	210	8	4	6
11	228	8	4	6
12	246	8	4	6
13	264	8	4	7
Z0 Z7
leng	n	dz	优化前dx	优化后dx 
9	192	8	2	4
10	210	8	4	4
11	228	8	2	6
13	264	8	3	7
