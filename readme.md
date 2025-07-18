# OSUREG: A registration package for multi-source geospatial data 

## Overview

`OSUREG` is registration package including state-of-the-art registration algorithm for handling multi-source geospatial data. It supports point cloud, DSM formats.

## Install

1. Clone the repo:

    ```console
    git clone https://github.com/Ggs1mida/OSUREG.git
    ```

2. Create and activate a Conda environment containing the required dependencies. From inside the `OSUREG` directory:

    ```console
    conda env create -n osureg python=3.11
    ```

    ```console
    conda activate osureg
    ```

3. Install dependencies via `install.txt`.

## CoRegistration

### Running OSUREG

```
osureg <reference data path> <moving data path> --type ['DSM','3D','G2F','G2A']
```
Mode  
"3D": support 3D point cloud registration, focus on small-scale data, and register them using 3D features  
"DSM": support 2.5D DSM registration, use the proposed iterative closest algorithm, designed for DSM format for lower computation and memory consumption.    
"G2F": support ground to building footprint (kml format) registration  
"G2A": support ground to air registration

