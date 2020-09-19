# Mapping the wetland regions of Colombia

**Purpose**: Build models to predict wetland areas in Colombia.

## Using this code (How-To video): https://www.youtube.com/watch?v=Xt-rrXzqVUk

## Link to rasters of predicted Mangroves in Colombia: https://drive.google.com/drive/folders/1C-5yrKwxzQecGLOMAJnMKWp00uD1z9K4?usp=sharing

Each predicted raster in the above Google Drive folder has two bands: class_id and confidence. class_id=1 designates predicted mangroves across all rasters. The confidence band is between 0 and 1 and it designates the confidence of a pixel belonging to the predicted class_id.

## Link to pre-downloaded data: https://drive.google.com/drive/folders/1GI9qM3yHtQxSf6828VCTQoFQ_GfcMdUZ?usp=sharing

At the above Google Drive folder you will find pre-downloaded data for the CGSM and Sanquianga regions. If using these, please set the ***ONLINE*** variable to False in Step 6 below and move these raster files into the "roi_data" directory.

# Steps to Use

1. Clone the repository.

2. In the command line, navigate to the repository and run *"pip install -r requirements.txt"*. This will download the needed python modules to run this code. The only other module you will need is *osgeo*.

3. Create a folder on your Google Drive called **GoogleEarthEngine**. Your data will be stored in this folder temporarily before being downloaded locally in batches. Set the ***GOOGLE_EARTH_ENGINE_GDRIVE_FOLDER_ID*** variable in the Jupyter notebook *"Predicting Colombia Wetlands.ipynb"*. You can get this by visiting your Google Drive, right clicking the folder, clicking "Get Link" and copying the unique id following "folders" in the link. *Make sure that the folder containing the data on Google Drive is public.*

![alt text](https://github.com/ritvikmath/ritvikmath.github.io/blob/master/images/2020-09-17_18-29-37.png)

4. Run the notebook *"Python Google Drive Setup.ipynb"*. This allows your python code to communicate with your Google Drive, where the Google Earth Engine data gets temporarily stored before being downloaded locally. You will know this step is successful if a file called "token.pickle" is created in your working directory. This file contains the needed information for the Python code to interact with your Google Drive.

5. Modify (add or remove polygons) the file at **roi_polygons/roi_polygons.shp** to contain polygons in which to predict wetlands. If you wish to run this notebook in demo mode (only predicts the CGSM and Sanquianga regions) please change the ***ROI_POLYGONS_FILE*** variable to "./roi_polygons/roi_polygons_limited.shp" in the Jupyter notebook *"Predicting Colombia Wetlands.ipynb"*. 

6. Set the following variables in the **User Input Area** of the Jupyter notebook *"Predicting Colombia Wetlands.ipynb"*:

***SELECTED_BANDS***: the bands that will be used in the prediction

***MAX_CONSIDERED_ELEVATION***: the maximum elevation (meters) to consider in the prediction process

***DATE_RANGE***: the start and end dates to consider in the prediction process

***METHOD***: the method you would like to use for prediction; the options are either "random_forest" or "histogram"

***ONLINE***: True if you wish to download data from Google Earth Engine (needs internet connection), False if you will use your own rasters (does not need internet connection). If using your own rasters, put them in the directory *roi_data*.

7. Run the driver code under the heading **Driver Code : Download ROI Polygon Data**. This will download data from Google Earth Engine for your ROI areas.

8. Run the driver code under the heading **Driver Code : Get Suggested Number of Classes**. This will cluster the roi data by various number of clusters, searching for the best separation between clusters. It will save files in the **roi_suggested**. These serve as rough guidelines in determining how many classes are needed for training data in the next step.

9. Using a GIS software like QGIS, use the suggested classes rasters as guidelines for this step. Within each polygon in **roi_polygons.shp**, create several small training polygons, each having a *class_id* and *confidence* between 0 and 1. This step allows the user to label small regions within each roi polygon, helping build a model to predict regions with similar signatures within the training region. Refer to the file at **training_polygons/training_polygons.shp** as a reference and modify this file by adding or removing training polygons. Try to include as many training polygons from each class as possible for a more robust prediction.

![alt text](https://github.com/ritvikmath/ritvikmath.github.io/blob/master/images/2020-09-17_18-32-04.png)

10. Run the driver code under the heading **Driver Code : Extract Training Data**. This will extract the data from your training polygons.

11. Run the driver code under the heading **Driver Code : Classify ROI Areas**. This will build the model and use it to predict the class_id of each pixel within your roi polygons. The results are stored in the **roi_predicted** folder. Each of these prediction rasters has two layers. The first is the predicted class id and the second is the confidence between 0 and 1 at each pixel. Additional rasters are stored in the **roi_gmw** folder; these rasters compare the predictd results to the Global Mangrove Watch (GMW) and each of these rasters contains values in {0,1,2} where 0 means the prediction marked this as mangrove but the GMW did not, 1 means that the GMW marked this as mangrove but the prediction did not, and 2 means that both the GMW and the prediction marked this as mangroves. These rasters are intended to visualize where the GMW and predictions agree and disagree.

![alt text](https://github.com/ritvikmath/ritvikmath.github.io/blob/master/images/2020-09-17_18-32-48.png)

