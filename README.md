# BinRNN
The Provenance Analysis of Binary Code via Recurrent Neural Networks

### Introduction

This is a reference program for the more advanced method `BinProv` to compare the experiemental results.

Both `BinRNN` and `BinProv` appear in the RAID 2020 paper: [BinProv: Binary Code Provenance Identification without Disassembly](https://dl.acm.org/doi/abs/10.1145/3545948.3545956)

* You can find `BinProv` on github: [Viewer-HX/BinProv](https://github.com/Viewer-HX/BinProv)

### Prerequirements

```
gc
numpy
torch
sklearn
```

### File Structure

``` shell
.
├── data # dataset.
├── logs # record the terminal outputs.
├── temp # store the models.
├── BinRNN.py # code.
└── README.md
```

### Usage

```shell
>> python3 BinRNN.py
```

### Citation
```bibtex
@inproceedings{xu2022binprov,
author = {He, Xu and Wang, Shu and Xing, Yunlong and Feng, Pengbin and Wang, Haining and Li, Qi and Chen, Songqing and Sun, Kun},
title = {BinProv: Binary Code Provenance Identification without Disassembly},
year = {2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the 25th International Symposium on Research in Attacks, Intrusions and Defenses},
pages = {350–363},
numpages = {14},
location = {Limassol, Cyprus},
series = {RAID '22}
}
```
