#### Introduction

This repo is part of a collaborative project aimed at classifying all 150,000 lakes in the continental US by the degree of human modification such as the presence of dams or water-control structures. 

#### Project Structure

- `cnn` - Tensorflow code
  - `data_load.py` - Data Loader
  - `tensorflow.py` - CNN based on pure Tensorflow
  - `tflearn.py` - CNN based on TFlearn
- `gee` - Google Earth Engine code for feature extraction
  - `exports` - Feature exports
  - `exports_iws` - Feature exports for iws
- `map_sampling` - Map partitioning
  - `main.py`
  - `samples.py`
- `ml` - Machine Learning based on GEE features
  - `models.ipynb`

