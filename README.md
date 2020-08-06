# Mapping the wetland regions of Colombia

**Purpose**: Build models to predict wetland areas in Colombia

# Steps to Use

## (1) DataDownload.ipynb

1. Download the baseline wetlands map of colombia at the Google Drive link:

https://drive.google.com/file/d/1GHmrHCjlvCJecxAGdkp8MG02i6aptyft/view?usp=sharing

Set the ***BASELINE_WETLAND_PATH*** to the path where you have stored this file locally

2. Choose a rectangular region within Colombia you wish to analyze. Note the ***(minx, miny, maxx, maxy)***. Fill those in under ***SET YOUR AREA OF INTEREST HERE***. Run the ***generate_region_folders*** function call.

3. Fill in all desired parameters under ***SET YOUR PARAMETERS HERE***.

4. Run the driver code and wait for all Google Earth Engine tasks to complete. Make sure that the folder containing the data on Google Drive is public.

**Expected Output**: files stored on Google Drive with the datasets and bands specified

## (2) CacheTrainingData.ipynb

1. Set the ***GOOGLE_EARTH_ENGINE_GDRIVE_FOLDER_ID*** You can get this by visiting your Google Drive, right clicking the folder where you stored the Google Earth Engine data, clicking "Get Shareable Link" and copying the unique id following "folders" in the link.

2. Follow the steps under the header ***Folder Name to File ID*** to automatically get the Google Drive file ids of all files downloaded in the last notebook. You can also manually populate the dictionary ***folder_name_to_file_id*** with pairs like "folder_name: file_id" but might be tedious if many folders.

3. Fill in the sub-type of wetland you wish to predict under ***CHOOSE THE WETLAND SUB-TYPE YOU WOULD LIKE TO PREDICT***.

4. Run the driver code and wait for it to complete.

**Expected Output**: a file in the working directory called ***stored_training_data.p*** containing the training data for all sub-regions

## (3) WetlandPrediction.ipynb

1. Run the drive code and wait for it to complete.

**Expected Output**: rasters in each sub-region directory starting with "predicted..." which contain the wetland sub-type predictions within that sub-region

## (4) PostProcessing.ipynb

1. Run the drive code and wait for it to complete.

**Expected Output**: rasters in each sub-region directory starting with "new_baseline..." which contain the updated baseline map for that sub-region.


