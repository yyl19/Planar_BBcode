# Planar Fault-Tolerant Quantum Computation with Low Overhead

This repository contains the code implementation accompanying the paper:

> **Planar Fault-Tolerant Quantum Computation with Low Overhead**  
> [arXiv:2506.18061](http://arxiv.org/abs/2506.18061)

The code focuses on generating planar quantum error-correcting codes and performing various logical Pauli measurements in a fault-tolerant manner using the proposed **planar BB code craft**. An example implementation for the [[162, 8, 7]] code is provided.

## 📂 Code Structure

| File            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `planarBBcode.py` | Generates the planar BB code structure, including `Hx`, `Hz` parity-check matrices. |
| `Xm.py`         | Performs **single-block** $X$ and $XX$ logical Pauli measurements.          |
| `Zm.py`         | Performs **single-block** $Z$ and $ZZ$ logical Pauli measurements.          |
| `jointx.py`     | Performs **two-block** $X$-type joint logical measurements.                 |
| `jointz.py`     | Performs **two-block** $Z$-type joint logical measurements.                 |
## 📂 Visualizations and Code Structure

The `plot` folder contains several illustrative figures:
- `54_180_xtile.png` and `54_180_ztile.png` visualize the **X- and Z-type stabilizers** of the code after tile layout.
- `54layout.png` shows the **original layout** of the code before deformation.

The `logical_basis` folder provides the **logical operator bases** of the original code.

The `MP` folder contains the **deformed version of the code** and the corresponding **optimized logical operators**, which are obtained by the painting procedure during the measurement of `P`.

## 🧪 Example: [[162, 8, 7]] Code
### 1. Generate the Code

```bash
$ python planarBBcode.py
The [[162,8,7]] code has been saved to './data/hxhz_7_7.pkl'.
```

### 2.Performs **two-block** $Z$-type joint logical measurements
```bash
$ python jointz.py
========= Summary of joint Z =========
Z2 tensor Z3
leng	n	dx	before opt dz	after dz 
24	468	7	3	7
```
### 3.Performs **two-block** $X$-type joint logical measurements
```bash
$ python jointx.py
========= Summary of joint X =========
X0 tensor X1
wid	n	dz	before opt dx	after opt dx 
25	486	7	3	7

```


### 4. Performs **single-block** $X$ and $XX$
```bash
$ python Xm.py
========= Summary X  =========
X0
wid	n	dz	Before opt dx	After opt dx 
9	190	7	2	3
10	208	7	2	4
11	226	7	3	5
12	244	7	6	7
X1
wid	n	dz	Before opt dx	After opt dx 
9	190	7	2	4
10	208	7	2	4
11	226	7	4	6
12	244	7	4	6
13	262	7	6	7
X2
wid	n	dz	Before opt dx	After opt dx 
9	190	8	2	2
10	208	8	2	3
11	226	8	2	6
12	244	8	3	6
13	262	8	3	6
14	280	8	3	6
15	298	8	3	7
X3
wid	n	dz	Before opt dx	After opt dx 
9	190	8	2	2
10	208	8	2	3
11	226	8	2	6
12	244	8	2	7
X4
wid	n	dz	Before opt dx	After opt dx 
9	190	8	2	3
10	208	8	2	4
11	226	8	2	6
12	244	8	2	6
13	262	8	3	6
14	280	8	2	7
X5
wid	n	dz	Before opt dx	After opt dx 
9	190	7	2	4
10	208	7	2	5
11	226	7	3	6
12	244	7	4	7
X6
wid	n	dz	Before opt dx	After opt dx 
9	190	7	2	4
10	208	7	2	5
11	226	7	3	6
12	244	7	3	6
13	262	7	3	7
X7
wid	n	dz	Before opt dx	After opt dx 
9	190	7	3	4
10	208	7	5	5
11	226	7	6	6
12	244	7	7	7

========= Summary XX =========
X0 X1
wid	n	dz	Before opt dx	After opt dx 
9	190	8	2	3
10	208	8	2	4
11	226	8	2	4
12	244	8	3	6
13	262	8	2	7
X0 X2
wid	n	dz	Before opt dx	After opt dx 
9	190	8	2	4
10	208	8	2	4
11	226	8	2	6
12	244	8	2	6
13	262	8	3	6
14	280	8	2	6
15	298	8	3	7
X0 X3
wid	n	dz	Before opt dx	After opt dx 
9	190	7	5	5
10	208	7	5	5
11	226	7	7	7
X0 X4
wid	n	dz	Before opt dx	After opt dx 
9	190	8	3	4
10	208	8	2	4
11	226	8	2	6
12	244	8	3	6
13	262	8	4	7
X0 X5
wid	n	dz	Before opt dx	After opt dx 
9	190	7	3	3
10	208	7	4	5
11	226	7	3	5
12	244	7	3	7
X0 X6
wid	n	dz	Before opt dx	After opt dx 
9	190	8	2	4
10	208	8	2	4
11	226	8	2	6
12	244	8	2	6
13	262	8	2	6
14	280	8	2	7
X0 X7
wid	n	dz	Before opt dx	After opt dx 
9	190	7	2	4
10	208	7	3	4
11	226	7	3	6
12	244	7	5	6
13	262	7	5	6
14	280	7	7	7
X1 X2
wid	n	dz	Before opt dx	After opt dx 
9	190	8	2	2
10	208	8	2	3
11	226	8	3	6
12	244	8	5	6
13	262	8	6	7
X1 X3
wid	n	dz	Before opt dx	After opt dx 
9	190	8	2	2
10	208	8	2	3
11	226	8	2	4
12	244	8	2	5
13	262	8	2	7
X1 X4
wid	n	dz	Before opt dx	After opt dx 
9	190	8	2	3
10	208	8	2	4
11	226	8	2	6
12	244	8	2	6
13	262	8	2	7
X1 X5
wid	n	dz	Before opt dx	After opt dx 
9	190	7	2	4
10	208	7	2	5
11	226	7	2	6
12	244	7	4	7
X1 X6
wid	n	dz	Before opt dx	After opt dx 
9	190	8	2	4
10	208	8	3	4
11	226	8	4	6
12	244	8	3	6
13	262	8	3	7
X1 X7
wid	n	dz	Before opt dx	After opt dx 
9	190	8	3	4
10	208	8	2	5
11	226	8	2	6
12	244	8	2	7
X2 X3
wid	n	dz	Before opt dx	After opt dx 
9	190	8	2	4
10	208	8	2	4
11	226	8	2	6
12	244	8	2	7
X2 X4
wid	n	dz	Before opt dx	After opt dx 
9	190	7	2	5
10	208	7	2	5
11	226	7	2	7
X2 X5
wid	n	dz	Before opt dx	After opt dx 
9	190	8	2	2
10	208	8	2	3
11	226	8	3	6
12	244	8	2	6
13	262	8	2	6
14	280	8	3	7
X2 X6
wid	n	dz	Before opt dx	After opt dx 
9	190	7	2	4
10	208	7	2	4
11	226	7	3	6
12	244	7	5	7
X2 X7
wid	n	dz	Before opt dx	After opt dx 
9	190	8	2	2
10	208	8	2	3
11	226	8	2	5
12	244	8	2	6
13	262	8	3	7
X3 X4
wid	n	dz	Before opt dx	After opt dx 
9	190	7	6	6
10	208	7	6	6
11	226	7	7	7
X3 X5
wid	n	dz	Before opt dx	After opt dx 
9	190	7	3	4
10	208	7	2	4
11	226	7	3	6
12	244	7	3	6
13	262	7	5	6
14	280	7	5	6
15	298	7	7	7
X3 X6
wid	n	dz	Before opt dx	After opt dx 
9	190	7	2	4
10	208	7	2	4
11	226	7	2	6
12	244	7	2	6
13	262	7	3	6
14	280	7	3	6
15	298	7	3	7
X3 X7
wid	n	dz	Before opt dx	After opt dx 
9	190	7	4	4
10	208	7	3	5
11	226	7	3	6
12	244	7	5	7
X4 X5
wid	n	dz	Before opt dx	After opt dx 
9	190	7	3	4
10	208	7	3	4
11	226	7	2	6
12	244	7	3	6
13	262	7	5	7
X4 X6
wid	n	dz	Before opt dx	After opt dx 
9	190	7	2	3
10	208	7	2	3
11	226	7	2	4
12	244	7	2	5
13	262	7	2	7
X4 X7
wid	n	dz	Before opt dx	After opt dx 
9	190	7	3	4
10	208	7	3	4
11	226	7	5	6
12	244	7	5	7
X5 X6
wid	n	dz	Before opt dx	After opt dx 
9	190	7	2	2
10	208	7	2	3
11	226	7	2	5
12	244	7	2	5
13	262	7	2	7
X5 X7
wid	n	dz	Before opt dx	After opt dx 
9	190	7	2	4
10	208	7	3	4
11	226	7	4	6
12	244	7	5	6
13	262	7	4	6
14	280	7	3	7
X6 X7
wid	n	dz	Before opt dx	After opt dx 
9	190	7	2	4
10	208	7	2	6
11	226	7	2	6
12	244	7	2	7
```
### 5. Performs **single-block** $Z$ and $ZZ$
```bash
$ python Zm.py
========= Summary Z =========
Z0
leng	n	dx	Before opt dz	After opt dz 
9	192	9	2	4
10	210	9	2	5
11	228	9	2	6
12	246	9	3	6
13	264	9	4	7
Z1
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	3
10	210	8	2	5
11	228	8	2	5
12	246	8	2	6
13	264	8	3	6
14	282	8	2	7
Z2
leng	n	dx	Before opt dz	After opt dz 
9	192	7	2	5
10	210	7	2	6
11	228	7	3	6
12	246	7	2	7
Z3
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	5
10	210	8	2	6
11	228	8	2	7
Z4
leng	n	dx	Before opt dz	After opt dz 
9	192	7	3	6
10	210	7	4	7
Z5
leng	n	dx	Before opt dz	After opt dz 
9	192	7	2	5
10	210	7	5	6
11	228	7	6	6
12	246	7	7	7
Z6
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	5
10	210	8	4	6
11	228	8	4	6
12	246	8	5	6
13	264	8	6	7
Z7
leng	n	dx	Before opt dz	After opt dz 
9	192	7	2	4
10	210	7	2	5
11	228	7	2	6
12	246	7	2	6
13	264	7	2	6
14	282	7	2	7

========= Summary ZZ =========
Z0 Z1
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	4
10	210	8	5	5
11	228	8	4	6
12	246	8	2	6
13	264	8	7	7
Z0 Z2
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	5
10	210	8	2	6
11	228	8	2	6
12	246	8	2	6
13	264	8	2	6
14	282	8	3	7
Z0 Z3
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	4
10	210	8	2	5
11	228	8	2	6
12	246	8	3	7
Z0 Z4
leng	n	dx	Before opt dz	After opt dz 
9	192	9	2	3
10	210	9	2	4
11	228	9	2	5
12	246	9	2	6
13	264	9	3	6
14	282	9	4	6
15	300	9	4	7
Z0 Z5
leng	n	dx	Before opt dz	After opt dz 
9	192	8	3	4
10	210	8	2	5
11	228	8	2	6
12	246	8	3	7
Z0 Z6
leng	n	dx	Before opt dz	After opt dz 
9	192	8	4	4
10	210	8	4	6
11	228	8	4	6
12	246	8	4	6
13	264	8	4	7
Z0 Z7
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	4
10	210	8	4	4
11	228	8	2	6
12	246	8	2	7
Z1 Z2
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	3
10	210	8	2	4
11	228	8	3	5
12	246	8	4	6
13	264	8	5	7
Z1 Z3
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	4
10	210	8	2	6
11	228	8	4	6
12	246	8	5	7
Z1 Z4
leng	n	dx	Before opt dz	After opt dz 
9	192	8	3	4
10	210	8	2	4
11	228	8	2	5
12	246	8	3	6
13	264	8	4	6
14	282	8	2	7
Z1 Z5
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	2
10	210	8	2	3
11	228	8	2	5
12	246	8	3	6
13	264	8	2	7
Z1 Z6
leng	n	dx	Before opt dz	After opt dz 
9	192	7	2	4
10	210	7	2	5
11	228	7	2	7
Z1 Z7
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	4
10	210	8	2	5
11	228	8	5	6
12	246	8	6	7
Z2 Z3
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	4
10	210	8	2	5
11	228	8	4	6
12	246	8	4	6
13	264	8	5	6
14	282	8	6	7
Z2 Z4
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	5
10	210	8	2	5
11	228	8	3	6
12	246	8	3	7
Z2 Z5
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	5
10	210	8	2	6
11	228	8	3	6
12	246	8	4	6
13	264	8	2	6
14	282	8	2	6
15	300	8	2	7
Z2 Z6
leng	n	dx	Before opt dz	After opt dz 
9	192	9	2	3
10	210	9	2	4
11	228	9	2	5
12	246	9	2	7
Z2 Z7
leng	n	dx	Before opt dz	After opt dz 
9	192	7	2	3
10	210	7	2	4
11	228	7	3	6
12	246	7	2	6
13	264	7	2	6
14	282	7	2	7
Z3 Z4
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	4
10	210	8	2	5
11	228	8	3	6
12	246	8	2	7
Z3 Z5
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	5
10	210	8	2	6
11	228	8	2	6
12	246	8	4	7
Z3 Z6
leng	n	dx	Before opt dz	After opt dz 
9	192	7	2	5
10	210	7	2	5
11	228	7	2	6
12	246	7	2	6
13	264	7	2	6
14	282	7	2	7
Z3 Z7
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	5
10	210	8	3	6
11	228	8	5	7
Z4 Z5
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	5
10	210	8	2	6
11	228	8	3	6
12	246	8	5	7
Z4 Z6
leng	n	dx	Before opt dz	After opt dz 
9	192	9	2	4
10	210	9	3	5
11	228	9	2	5
12	246	9	2	6
13	264	9	3	7
Z4 Z7
leng	n	dx	Before opt dz	After opt dz 
9	192	8	2	5
10	210	8	2	5
11	228	8	2	6
12	246	8	3	7
Z5 Z6
leng	n	dx	Before opt dz	After opt dz 
9	192	7	3	5
10	210	7	3	6
11	228	7	2	6
12	246	7	2	7
Z5 Z7
leng	n	dx	Before opt dz	After opt dz 
9	192	7	2	6
10	210	7	2	7
Z6 Z7
leng	n	dx	Before opt dz	After opt dz 
9	192	8	4	5
10	210	8	2	6
11	228	8	2	6
12	246	8	3	7

```

