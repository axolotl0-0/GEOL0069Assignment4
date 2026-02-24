# Echoes in Leads and Sea Ice Classification using Sentinel-2 optical data and Sentinel-3 altimetry data.
GEOL0069 AI4EO Assignment 4
Student Number: 22170262

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#aims">Aims</a></li>
    <li><a href="#getting-started---prerequisites--installation">Getting Started - Prerequisites + Installation</a></li>
    <li>
      <a href="#k-means-and-gmm">K-Means and GMM</a>
      <ul>
        <li><a href="#k-means-clustering">K-Means Clustering</a></li>
        <li><a href="#gaussian-mixture-models-gmm">Gaussian Mixture Models (GMM)</a></li>
      </ul>
    </li>
    <li>
      <a href="#notebook-roadmap">Notebook Roadmap</a>
      <ul>
        <li><a href="#image-classification---sentinel-2-imagery">Image Classification - Sentinel-2 Imagery</a></li>
        <li><a href="#altimetry-classification---sentinel-3-dataset">Altimetry Classification - Sentinel-3 Dataset</a></li>
        <li><a href="#plots-of-mean-and-standard-deviation-for-each-class">Plots of mean and standard deviation for each class</a></li>
        <li><a href="#waveform-aligment">Waveform Aligment</a></li>
        <li><a href="#comparison-with-esa-data">Comparison with ESA Data</a></li>
        <li><a href="#results-and-conclusions">Results and Conclusions</a></li>
      </ul>
    </li>
    <li><a href="#references">References</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About This Project

Radar echoes from ice leads, are narrow and sharply peaking waveforms, which can be distinguished from the broader waveforms of sea ice. This project uses this to identify ice leads and to hence classify them according to their waveform, and vice versa for sea ice.

This project first uses the Sentinel-2 optical data for unsupervised learning, using K-means clustering and Gaussian Mixture Models. The Sentinel-3 dataset is then used for altimetry classification, this is using the GMM model. The mean and standard deviation are hence found for each class, and the echoes are plotted.

The data obtained is then compared with the ESA data, using a confusion matrix and classification metrics to check precision.

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
#### Sentinel Data must also be downloaded
* These files include the following:
  ```sh
  Sentinel 2 Data
  S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/
  Sentinel 3 Data
  S3A_SR_2_LAN_SI_20190307T005808_20190307T012503_20230527T225016_1614_042_131______LN3_R_NT_005.SEN3
  ```
  
# K-Means and GMM
## K-Means Clustering
K-means clustering [1] is a type of unsupervised learning algorithm used for partitioning a dataset into a set of k groups (or clusters), where k represents the number of groups pre-specified by the analyst. It classifies the data points based on the similarity of the features of the data [2]. The basic idea is to define k centroids, one for each cluster, and then assign each data point to the nearest centroid, while keeping the centroids as small as possible.

### Why K-means for Clustering?

K-means clustering is particularly well-suited for applications where:

*    **The structure of the data is not known beforehand**: K-means doesn’t require any prior knowledge about the data distribution or structure, making it ideal for exploratory data analysis.
*    **Simplicity and scalability**: The algorithm is straightforward to implement and can scale to large datasets relatively easily.

### Key Components of K-means
*   **Choosing K:** The number of clusters (k) is a parameter that needs to be specified before applying the algorithm.
*   **Centroids Initialization:** The initial placement of the centroids can affect the final results.
*   **Assignment Step:** Each data point is assigned to its nearest centroid, based on the squared Euclidean distance.
*    **Update Step:** The centroids are recomputed as the center of all the data points assigned to the respective cluster.

The assignment and update steps are repeated iteratively until the centroids no longer move significantly, meaning the within-cluster variation is minimised. This iterative process ensures that the algorithm converges to a result, which might be a local optimum.

### Advantages of K-means
- **Efficiency**: K-means is computationally efficient.
- **Ease of interpretation**: The results of k-means clustering are easy to understand and interpret.

## Gaussian Mixture Models (GMM)
Gaussian Mixture Models (GMM) are a probabilistic model for representing normally distributed subpopulations within an overall population. The model assumes that the data is generated from a mixture of several Gaussian distributions, each with its own mean and variance [3],[4]. GMMs are widely used for clustering and density estimation, as they provide a method for representing complex distributions through the combination of simpler ones.

### Why Gaussian Mixture Models for Clustering?
Gaussian Mixture Models are particularly powerful in scenarios where:
*   **Soft clustering is needed:** Unlike K-means, GMM provides the probability of each data point belonging to each cluster, offering a soft classification and understanding of the uncertainties in our data.
*   **Flexibility in cluster covariance:** GMM allows for clusters to have different sizes and different shapes, making it more flexible to capture the true variance in the data.

### Advantages of GMM

*    **Soft Clustering**: Provides a probabilistic framework for soft clustering, giving more information about the uncertainties in the data assignments.
*    **Cluster Shape Flexibility**: Can adapt to ellipsoidal cluster shapes, thanks to the flexible covariance structure.

# Notebook Roadmap
### Basic Code Implementation
*    **K-means Clustering** - Introductory code for K-Means Clustering
*   **GMM** - A basic implementation of the Gaussian Mixture Model. This serves as an initial guide for understanding the model and applying it to the later data analysis.

## Image Classification - Sentinel-2 Imagery
We use these unsupervised machine learning methods to apply them to classification tasks focusing specifically on distinguishing between sea ice and leads in Sentinel-2 imagery. The optical brightness differences seperate sea ice and leads.
*   **K-means Clustering Implementation** - Code implements K-means clustering on the Sentinel-2 Bands.
*   **GMM Implementation** - GMM Implementation on Sentinel 2 data.

## Altimetry Classification - Sentinel-3 Dataset
This is the application of the previous unsupervised methods to altimetry classification tasks, focusing specifically on distinguishing between sea ice and leads in Sentinel-3 altimetry dataset. This data is radar data, using the peakiness (how sharp the radar return is) the data is able to be seperated into leads (sharp peak) and sea ice (broader).

#### Read in Functions Needed
Prior to the modeling process, it's crucial to preprocess the data to ensure compatibility with the analytical models. This involves transforming the raw data into variables, such as peakniness and stack standard deviation (SSD), etc. The functions needed for this are below:
* Functions
  ```sh
  from netCDF4 import Dataset
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.interpolate import griddata
  import numpy.ma as ma
  import glob
  from matplotlib.patches import Polygon
  import scipy.spatial as spatial
  from scipy.spatial import KDTree
  from sklearn.cluster import KMeans, DBSCAN
  from sklearn.preprocessing import StandardScaler,MinMaxScaler
  from sklearn.mixture import GaussianMixture
  from scipy.cluster.hierarchy import linkage, fcluster
  ```
#### Filtering the data
We filter the surface types to keep only sea ice and leads, and to remove other types.

#### Removing NaN values from data
There are some NaN values in the dataset so one way to deal with this is to delete them.

### Running the GMM model 
The code below runs the GMM Model, however it is also possible to subsitute for K-Means model or another preferred model.
* Code
  ```sh
  gmm = GaussianMixture(n_components=2, random_state=0)
  gmm.fit(data_cleaned)
  clusters_gmm = gmm.predict(data_cleaned)
  ```
## Plots of mean and standard deviation for each class
We then plot the mean waveform and standard deviation for each class
* Code for mean and standard deviation
  ```sh
  mean_ice = np.mean(waves_cleaned[clusters_gmm==0],axis=0)
  std_ice = np.std(waves_cleaned[clusters_gmm==0], axis=0)

  plt.plot(mean_ice, label='ice')
  plt.fill_between(range(len(mean_ice)), mean_ice - std_ice, mean_ice + std_ice, alpha=0.3)

  mean_lead = np.mean(waves_cleaned[clusters_gmm==1],axis=0)
  std_lead = np.std(waves_cleaned[clusters_gmm==1], axis=0)

  plt.plot(mean_lead, label='lead')
  plt.fill_between(range(len(mean_lead)), mean_lead - std_lead, mean_lead + std_lead, alpha=0.3)

  plt.title('Plot of mean and standard deviation for each class')
  plt.legend()
  ```
### Plot Echos
We can then plot all the echos, as well as the echos for the lead cluster and the sea ice cluster.
### Scatter Plots of Clustered Data
This  visualizes the clustering results using scatter plots, where different colors represent different clusters (`clusters_gmm`).

## Waveform Aligment
### Physical Waveform Alignment
To compare waveforms on a common footing we can **align** them using the known orbit geometry. This physically-based approach was developed at the Alfred Wegener Institute (AWI) [5]

### Effect of alignment on individual waveforms
We then look at the effect of alignment on individual waveforms, where we compare before and after the alignment.

### Aggregate alignment comparison
This code produces a figure which summarises the alignment effect across all waveforms.
The following are produced:
*    **Echogram** **before** and **after** alignment. A tighter bright band indicates better alignment.
*    **Histogram** of **peak** positions. The aligned distribution (red) should be narrower.
*    **Mean waveform per class**. After alignment the mean leading edge becomes sharper because individual waveforms are better registered.

## Comparison with ESA data
We then compare the results obtained with the ESA dataset.

In the ESA dataset, sea ice = 1 and lead = 2. Therefore, we need to subtract 1 from it so our predicted labels are comparable with the official product labels.
* Code:
  ```sh
  flag_cleaned_modified = flag_cleaned - 1
  from sklearn.metrics import confusion_matrix, classification_report
  true_labels = flag_cleaned_modified   # true labels from the ESA dataset
  predicted_gmm = clusters_gmm          # predicted labels from GMM method
  ```
### Quantifying the echo classification
Finally we quantify the echo classification against the ESA official classification using a confusion matrix.
* Code:
  ```sh
  # Compute confusion matrix
  conf_matrix = confusion_matrix(true_labels, predicted_gmm)

  # Print confusion matrix
  print("Confusion Matrix:")
  print(conf_matrix)

  # Compute classification report
  class_report = classification_report(true_labels, predicted_gmm)

  # Print classification report
  print("\nClassification Report:")
  print(class_report)
  ```
## Results and Conclusions
As the classification gave 22 false postives, and 24 false negatives, this means that out of 12195 samples there were only 46 errors. This suggests high precision of the model in classifying sea ice vs leads. Furthermore the f1 score is between 0.99-1.00 which suggests the model successfully finds almost (/all) real examples of each class. However there is also class imbalance of sea ice vs leads, but this is to be expected given the physical data.

In conclusion the Gaussian Mixture Model successfully separates sea ice and leads, achieving high classification similarity with ESA surface type flags.

# References
[1] Pattern recognition and machine learning. Christopher M. Bishop. 2006.

[2] SOME METHODS FOR CLASSIFICATION AND ANALYSIS OF MULTIVARIATE OBSERVATIONS. J. MACQUEEN. 1967

[3] Reynolds, D. (2009). Gaussian Mixture Models. Encyclopedia of Biometrics, pp.659–663. doi:https://doi.org/10.1007/978-0-387-73003-5_196.

[4] McLachlan, G. and Peel, D. (2004) Finite Mixture Models. Wiley, Hoboken.

[5] Alfred Wegener Institute (AWI) Physical Waveform Alignment
https://gitlab.awi.de/siteo/aligned-waveform-generator

# Contact
Serena Trant - serena.trant.23@ucl.ac.uk - 22170262

Project link: https://github.com/axolotl0-0/GEOL0069Assignment4

# Acknowledgments
Code was modified from code provided during GEOL0069 AI4EO as part of Assignment 4.











