# Mapping the wetland regions of Colombia

**Purpose**: Build models to predict wetland areas in Colombia.

# Steps to Use

1. Clone the repository.

1. Modify (add or remove polygons) the file at **prediction_polygons/prediction_polygons.shp** to contain polygons in which to predict wetlands. 

2. Choose which bands will be used in the prediction by setting ***SELECTED_BANDS*** variable in the Jupyter Notebook.

3. Set the ***MAX_CONSIDERED_ELEVATION*** variable to the maximum elevation (meters) pixels to consider in the prediction process.

4. Set the ***DATE_RANGE*** variable to the start and end dates to consider in the prediction process.

5. Set the ***METHOD*** variable to the method you would like to use for prediction. The options are either "random_forest" or "histogram".

6. Create a folder on your Google Drive called **GoogleEarthEngine**. Your data will be stored in this folder temporarily before being downloaded locally in batches. Set the ***GOOGLE_EARTH_ENGINE_GDRIVE_FOLDER_ID***. You can get this by visiting your Google Drive, right clicking the folder, clicking "Get Shareable Link" and copying the unique id following "folders" in the link. *Make sure that the folder containing the data on Google Drive is public.*

7. Follow the setup steps at https://developers.google.com/drive/api/v3/quickstart/python. This allows your Python code to interact with your Google Drive. Your token.pickle should be stored in the directory where you cloned this repository.

8. Run the driver code under the heading **Driver Code : Download Prediction Regions**. This will download data from Google Earth Engine for your prediction regions.

9. Run the driver code under the heading **Driver Code : Get Suggested Number of Classes**. This will cluster the prediction data by various bands and various number of clusters, searching for the best separation between clusters. It will save files in the **prediction_data** folder called **features_x_suggested.tiff**. These serve as rough guidelines in determining how many classes are needed for training data in the next step.

10. Using a GIS software like QGIS, use the **features_x_suggested.tiff** rasters as guidelines for this step. Within each polygon in **prediction_polygons.shp**, create several small training polygons, each having a *class_id* and *confidence* between 0 and 1. This step allows the user to label small regions within each prediction polygon, helping build a model to predict regions with similar signatures within the training region. Refer to the file at **training_polygons/training_polygons.shp** as a reference and modify this file by adding or removing training. Try to include as many training polygons from each class as possible for a more robust prediction.

11. Run the driver code under the heading **Driver Code : Download Training Data**. This will download all training data from Google Earth Engine and build a mapping between prediction polygons and the training_polygons that they contain.

12. Run the driver code under the heading **Driver Code : Classify Prediction Regions**. This will build the model and use it to predict the class_id of each pixel within your prediction polygons. The end result is files like **features_x_predicted.tiff** in the **prediction_data** folder. Each of these prediction rasters has two layers. The first is the predicted class id and the second is the confidence between 0 and 1 at each pixel.

