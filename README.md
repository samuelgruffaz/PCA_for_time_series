### PCA for time series

Authors: Samuel Gruffaz, Thibaut Germain


This repository gathers the functions developed in the paper [**“Shape Analysis for Time Series”**](https://proceedings.neurips.cc/paper_files/paper/2024/file/ad86418f7bdfa685cd089e028efd75cd-Paper-Conference.pdf), located in the `src` directory.

It is possible to represent **irregularly sampled time series of different lengths** and to apply **kernel PCA** to these representations in order to identify the main modes of shape variation in the time series.

These methods work particularly well when the analyzed dataset is **homogeneous in terms of shapes**, for example when each time series corresponds to:

* a heartbeat recording,
* a respiratory cycle,
* an electricity consumption pattern,
* or a heating load curve.

The `Docs` directory contains the files used to build the package documentation.

The `pages` directory contains the pages used to launch a **Streamlit application** from the menu, allowing users to test the different building blocks of the code.

**Coming next:**

* A class that combines the essential functions to simplify the user experience
* Complete documentation
* A PyPI release
* New kernels



