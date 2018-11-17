# Multidimensional segment trees can do range queries and updates in logarithmic time

### Introduction

This is the code for the paper :

Multidimensional segment trees can do range queries and updates in logarithmic time

Nabil Ibtehaz, Mohammad Kaykobad, Mohammad Sohel Rahman

[paper link](https://www.researchgate.net/publication/328758294_Multidimensional_segment_trees_can_do_range_queries_and_updates_in_logarithmic_time)

The webpage for this project is available [here](). 

A blogpost on this project is available [here](). 


### Overview

Here we present our implementation of the proposed 2D Segment Tree.

Currently we are only giving the python implementation. However in future we are interested in providing a C++ implementation as well.


Along with the code for dynamic range sum query. We have also provided the codes for multiplication, AND and OR queries as well. In future codes of more operations will hopefully be added in this repository.

### Instructions

The [rangeQuery.py](https://github.com/robin-0/Multidimensional-Segment-Tree/blob/master/python/sumQuery.py) code is sufficiently well documented. Please refer to that code, or the pseudocode in the paper to overcome any confusions regarding the implementation.

All the code shares the following:

1. [class](https://github.com/robin-0/Multidimensional-Segment-Tree/blob/master/python/sumQuery.py) SegmentTree2D : The proposed 2D Segment Tree implementation for solving the problem.


2. [class](https://github.com/robin-0/Multidimensional-Segment-Tree/blob/master/python/sumQuery.py) BruteForce : A brute force algorithm to solve the problem.


3. [function](https://github.com/robin-0/Multidimensional-Segment-Tree/blob/master/python/sumQuery.py) timingSimulation : Function for experimentally observing the time required for the proposed 2D Segment Tree operations.


4. [function](https://github.com/robin-0/Multidimensional-Segment-Tree/blob/master/python/sumQuery.py) simulation : Function to experimentally verify the correctness of the algorithm. We perform some random update and query operations. The identical operations are performed both using the brute force algorithm and our proposed algorithm. If the results differ by a threshold (1e-5), an error will occur.

### Requirements

The codes are written in raw python3. Only numpy was used to generate the random numbers
No additional packages are required.

### Cite

If you find this code useful in your research, please, consider citing our paper:

>@article{ibtehaz2018multidimensional,\
>  title={Multidimensional segment trees can do range queries and updates in logarithmic time},\
>  author={Ibtehaz, Nabil and Kaykobad, M and Rahman, M Sohel},\
>  journal={arXiv preprint arXiv:1811.01226},\
>  year={2018}\
>}
