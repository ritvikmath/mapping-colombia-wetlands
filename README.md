# Mapping the wetland regions of Colombia

Purpose: Build models to correct the baseline wetland maps of Colombia

# How to use

## Order of notebooks

Data Download -> Cache Training Data -> Machine Learning -> Post Processing

## (1) Data Download

This notebook allows the user to specify an area of interest and a set of features (Sentinel-2 and/or ALOS PALSAR). It then downloads the corresponding data from Google Earth Engine and puts it in the user's Google Drive.

## (2) Cache Training Data

This notebook downloads the data now in Google Drive and extracts information about the various bands at wetland sites and non-wetland sites. It stores all training data locally in a pickle (.p) file

## (3) Machine Learning

This notebook uses the locally cached pickle file to predict a specific wetland sub-type for the area of interest. Results are stored in TIFF raster files

## (4) Post Processing

This notebook takes the predicted wetlands and extracts "novel" wetlands. That is, wetlands that do not appear in the baseline but which are confidently predicted by the machine learning method. It produces an updated baseline map with these novel wetlands.
