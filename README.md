# Echoes in Leads and Sea Ice Classification using Sentinel-2 optical data and Sentinel-3 altimetry data.
GEOL0069 AI4EO Assignment 4
Student Number: 22170262


## Aims

The aim of this assigment is to: 
>• Classify the echoes in leads and sea ice and produce an average echo shape as well as standard deviation for these two classes.

>• Quantify the echo classification against the ESA official classification using a confusion matrix.


## Getting Started - Prerequisites + Installation


### Prerequisites

In order to complete the data processing the code needs to be mounted onto Google Drive.
* Google Drive
  ```sh
  from google.colab import drive
  drive.mount('/content/drive')
  ```
### Installation
#### The following libaries need to be installed:
* Rasterio and NetCDF
  ```sh
  pip install rasterio
  pip install netCDF4
  ```
#### Sentinel
# K-Means and GMM
## K-Means Clustering [1]
K-means clustering is a type of unsupervised learning algorithm used for partitioning a dataset into a set of k groups (or clusters), where k represents the number of groups pre-specified by the analyst. It classifies the data points based on the similarity of the features of the data {cite}macqueen1967some. The basic idea is to define k centroids, one for each cluster, and then assign each data point to the nearest centroid, while keeping the centroids as small as possible.

### Key Components of K-means
*   Choosing K: The number of clusters (k) is a parameter that needs to be specified before applying the algorithm.
*   Centroids Initialization: The initial placement of the centroids can affect the final results.
*   Assignment Step: Each data point is assigned to its nearest centroid, based on the squared Euclidean distance.
*   Update Step: The centroids are recomputed as the center of all the data points assigned to the respective cluster.

The assignment and update steps are repeated iteratively until the centroids no longer move significantly, meaning the within-cluster variation is minimised. This iterative process ensures that the algorithm converges to a result, which might be a local optimum.

## Gaussian Mixture Models (GMM)
Gaussian Mixture Models (GMM) are a probabilistic model for representing normally distributed subpopulations within an overall population. The model assumes that the data is generated from a mixture of several Gaussian distributions, each with its own mean and variance {cite}reynolds2009gaussian, mclachlan2004finite. GMMs are widely used for clustering and density estimation, as they provide a method for representing complex distributions through the combination of simpler ones.

# Notebook Roadmap
## Basic Code Implementation
### K-means Clustering
Introductory code for K-Means Clustering
### GMM
A basic implementation of the Gaussian Mixture Model. This serves as an initial guide for understanding the model and applying it to the later data analysis.

## Image Classification - Sentinel-2 imagery
We use these unsupervised machine learning methods to apply them to classification tasks focusing specifically on distinguishing between sea ice and leads in Sentinel-2 imagery.
### K-means Clustering Implementation
Code implements K-means clustering on the Sentinel-2 Bands.
### GMM Implementation
GMM Implementation on Sentinel 2 data.










# References
[1] Pattern recognition and machine learning. Christopher M. Bishop. 2006.

[2] SOME METHODS FOR CLASSIFICATION AND ANALYSIS OF MULTIVARIATE OBSERVATIONS. J. MACQUEEN. 1967

Alfred Wegener Institute (AWI) Physical Waveform Alignment
https://gitlab.awi.de/siteo/aligned-waveform-generator

# Contact
Serena Trant - serena.trant.23@ucl.ac.uk - 22170262

Project link: https://github.com/axolotl0-0/GEOL0069Assignment4

# Acknowledgments
Code was modified from code provided during GEOL0069 AI4EO as part of Assignment 4.











