=====================================
CSH : Compressed-Sensing for Herschel
=====================================

What is csh ?
==============

This is a package to test compressed-sensing methods for PACS/Herschel
data. 

Modules
=======
The structure of the package is as follows :

- compression : implements various compression matrices
- filter : utilities to derive noise filter from data (very alpha stage)
- mapmaking : calls all the mapmaking stages for a set of data
- score : methods to score various compression matrices (alpha)

Requirements
=============

List of requirements :

- numpy
- scipy
- tamasis (to be release : for PACS map-making)
- fht (for fast hadamard transform)
- lo (linear operators and sparse algorithms)

