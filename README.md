# Mapping the wetland regions of Colombia

**Purpose**: Build models to predict wetland areas in Colombia.

# Steps to Use

## (1) DataDownload.ipynb

1. Download the baseline wetlands map of Colombia at the Google Drive link:

https://drive.google.com/file/d/1GHmrHCjlvCJecxAGdkp8MG02i6aptyft/view?usp=sharing

Set the ***BASELINE_WETLAND_PATH*** to the path where you have stored this file locally.

2. Using a GIS software like QGIS, create a shapefile (.shp) containing polygons of areas you are interested in predicting. Name this file after the sub-type of wetland you wish to predict. An example called **mangrove.shp** is provided in this repository within the folder **area_of_interest**. Set the ***AREA_OF_INTEREST_FILE*** to the path where you have stored this file locally.

3. Create a folder on your Google Drive called **GoogleEarthEngine**. Your data will be stored in this folder temporarily before being downloaded locally in batches. Set the ***GOOGLE_EARTH_ENGINE_GDRIVE_FOLDER_ID***. You can get this by visiting your Google Drive, right clicking the folder, clicking "Get Shareable Link" and copying the unique id following "folders" in the link. *Make sure that the folder containing the data on Google Drive is public.*

4. Follow the setup steps under **Google Drive Access Instructions**. This allows your Python code to interact with your Google Drive.

5. Set the ***MAX_CONSIDERED_ELEVATION***. Any pixels above this elevation are disregarded in the following steps. This acts as a pre-processing data mask. For example, if predicting mangroves, you might set this as something like 6 meters.

6. Set the ***SELECTED_BANDS***. These are the bands that get used in the prediction. Note that the more bands that are chosen, the more space all training data will take up locally and the longer all subsequent prediction steps will take.

7. Run the driver code and wait for all Google Earth Engine tasks to complete.

**Expected Output**: all training data will be stored locally.

## (2) CacheTrainingData.ipynb

1. Run the driver code and wait for it to complete.

**Expected Output**: a file in the working directory called ***stored_training_data.p*** containing the training data for all sub-regions

## (3) WetlandPrediction.ipynb

1. Run the driver code and wait for it to complete.

**Expected Output**: rasters in each sub-region directory starting with "predicted..." which contain the wetland sub-type predictions within that sub-region

## (4) PostProcessing.ipynb

1. Set the ***PERCENTILE_THRESHOLD***. This number, between 0 and 1, contols how sensitive the classifier will be in identifying new wetlands and removing existing wetlands from the baseline. A higher number will result in a more conservative approach (less change from baseline but higher confidence in those changes) while a lower number results in a more liberal approach (more change from the baseline but lower confidence in those changes).

2. Run the driver code and wait for it to complete.

**Expected Output**: rasters in each sub-region directory starting with "new_baseline..." which contain the updated baseline map for that sub-region; a interactive map called map.html will be generated to view the predicted wetlands in context


