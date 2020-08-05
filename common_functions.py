from osgeo import gdal, osr
import numpy as np
from scipy.signal import convolve
import os

NO_DATA_VALUE = 65535

feature_to_code = {'None': 0, 
                   'Open Water': 10, 
                   'Mangrove': 20, 
                   'Swamp / Bog': 30, 
                   'Fen': 40, 
                   'Riverine': 50, 
                   'Floodswamp': 60, 
                   'Floodplain': 70, 
                   'Marsh': 80, 
                   'Wetland in dry areas': 90, 
                   'Wet meadow': 100}

def get_scale(fine_res_ds, coarse_res_ds, reshape_ds=False):
    """
    fine_res_ds: the finer resoultion GDAL dataset
    coarse_res_ds: the coaser resoultion GDAL dataset
    reshape_ds: True if we wish to save files with datasets of divisible proportions
    """
    
    #get the geotransforms
    gt_fine = fine_res_ds.GetGeoTransform()
    gt_coarse = coarse_res_ds.GetGeoTransform()
    
    #try to get the scale of the coarse to fine datasets
    exact_scale = gt_coarse[1] / gt_fine[1]
    int_scale, remainder = int(exact_scale), exact_scale%1
    
    #if asked to save new files
    if reshape_ds:
        fine_res_arr = fine_res_ds.ReadAsArray()
        _, height_fine, width_fine = fine_res_arr.shape
        
        height_fine = int_scale * (height_fine // int_scale)
        width_fine = int_scale * (width_fine // int_scale)
        
        #save new fine dataset
        options_fine = gdal.WarpOptions(format='GTiff', width=width_fine, height=height_fine, srcNodata=NO_DATA_VALUE, dstNodata=NO_DATA_VALUE)
        ds = gdal.Warp(destNameOrDestDS='rescaled_fine.tiff', srcDSOrSrcDSTab=fine_res_ds, options=options_fine)
        ds = None
        
        height_coarse = height_fine // int_scale
        width_coarse = width_fine // int_scale
        
        #save new coarse datset
        options_corase = gdal.WarpOptions(format='GTiff', width=width_coarse, height=height_coarse)
        ds = gdal.Warp(destNameOrDestDS='rescaled_coarse.tiff', srcDSOrSrcDSTab=coarse_res_ds, options=options_corase)
        ds = None
            
    return int_scale, remainder

def split(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size
    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w, k = arr.shape
    band_splits = []
    for i in range(k):
        curr_band_split = arr[:,:,i].reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols)
        band_splits.append(curr_band_split)
    return np.stack(band_splits, axis=-1)

def preprocess_data_set_pair(ds_features, ds_labels, wetland_type):
    """
    ds_train: the training dataset 
    ds_test: the testing dataset
    wetland_type: the sub-type of wetland such as "Mangrove" or "Marsh"
    """
    
    #preprocess to make sure that train data sets have compatible dimensions
    scale, _ = get_scale(ds_features, ds_labels, True)
    
    ds_features = gdal.Open('rescaled_fine.tiff', gdal.GA_ReadOnly)
    arr_features = ds_features.ReadAsArray()
    arr_features = np.stack(arr_features, axis=-1)
    gt = ds_features.GetGeoTransform()
    ds_features = None
    
    ds_labels = gdal.Open('rescaled_coarse.tiff', gdal.GA_ReadOnly)
    arr_labels = ds_labels.ReadAsArray()
    ds_labels = None
    
    #split the features array into chunks to be compatible with the labels array
    labels_h, labels_w = arr_labels.shape
    features_h, features_w, features_k = arr_features.shape
    split_arr_features = split(arr_features, scale, scale).reshape(labels_h, labels_w, scale, scale, features_k)
    
    #get target value
    tgt_val = feature_to_code[wetland_type]
    
    #assign labels
    arr_labels = arr_labels.astype(float)
    arr_labels[arr_labels == tgt_val] = 1
    arr_labels[(arr_labels == 255)|(arr_labels == 10)] = -1
    arr_labels[abs(arr_labels) != 1] = 0
    
    return arr_labels, split_arr_features, gt

def create_raster(output_path, columns, rows, nband=1, gdal_data_type=gdal.GDT_Int32, driver=r'GTiff'):
    ''' 
    returns gdal data source raster object 
    '''
    
    # create driver
    driver = gdal.GetDriverByName(driver)

    output_raster = driver.Create(output_path, columns, rows, nband, eType = gdal_data_type)    
    
    return output_raster

def np_array_to_raster(output_path, arr, geotransform, no_data=None, nband=1, gdal_data_type=gdal.GDT_Int32, spatial_reference_system_wkid=4326, driver=r'GTiff'):
    ''' 
    returns a gdal raster data source

    keyword arguments:

    output_path -- full path to the raster to be written to disk
    numpy_array -- numpy array containing data to write to raster
    upper_left_tuple -- the upper left point of the numpy array (should be a tuple structured as (x, y))
    cell_resolution -- the cell resolution of the output raster
    no_data -- value in numpy array that should be treated as no data
    nband -- the band to write to in the output raster
    gdal_data_type -- gdal data type of raster (see gdal documentation for list of values)
    spatial_reference_system_wkid -- well known id (wkid) of the spatial reference of the data
    driver -- string value of the gdal driver to use
    '''

    rows, columns = arr.shape[0], arr.shape[1]

    # create output raster
    output_raster = create_raster(output_path, columns, rows, nband, gdal_data_type) 

    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(spatial_reference_system_wkid)
    output_raster.SetProjection(spatial_reference.ExportToWkt())
    output_raster.SetGeoTransform(geotransform)
    
    for band_idx in range(1,nband+1):
        output_band = output_raster.GetRasterBand(band_idx)
        if no_data != None:
            output_band.SetNoDataValue(no_data)
        if nband > 1:
            output_band.WriteArray(arr[:,:,band_idx-1])
        else:
            output_band.WriteArray(arr)
        output_band.FlushCache() 
        output_band.ComputeStatistics(False)

    if os.path.exists(output_path) == False:
        raise Exception('Failed to create raster: %s' % output_path)

    return output_raster

def apply_convolution(img, kernel):
    """
    convolve the given image with the given kernel, maintaining the original image dimensions|
    """
    
    #convolve blur kernel with image
    convolvedImg = convolve(img, kernel, mode='same')
    
    return(convolvedImg)

def sample_by_vicinity_scores(pairs, vicinity_scores, thresh):
    """
    pairs: a double of xcoords and ycoords 
    vicinity_scores: scores between 0 and thresh, where lower means more confident
    thresh: the upper bound for the vicinity scores
    """
    
    probs = (thresh - vicinity_scores)
    probs = probs / np.sum(probs)
    
    all_indices = np.arange(len(vicinity_scores))
    chosen_indices = np.random.choice(all_indices, len(all_indices), p=probs)
    chosen_pairs = (pairs[0][chosen_indices], pairs[1][chosen_indices])
    
    return chosen_pairs

def standardize_data(arr, means, devs, num_sd_trim):
    """
    arr: input array (N samples x k features)
    means: the column means to use in normalization
    devs: the column deviations to use in normalization
    num_sd_trim: the number of standard deviations outside of which to remove values in arr
    """
    
    arr_norm = (arr - means) / devs
    arr_norm[arr_norm > num_sd_trim] = num_sd_trim
    arr_norm[arr_norm < -num_sd_trim] = -num_sd_trim
    
    return arr_norm

def get_approx_densities(arr, approx_vals, num_sd_trim=2.5, bins=10):
    """
    arr: input array (N samples x k features)
    approx_vals: the values whose densities to approximate  (M samples x k features)
    """
    
    #get data properties
    N,k = arr.shape
    stride = 2*num_sd_trim / bins
    bin_edges = np.arange(-num_sd_trim,num_sd_trim+stride,stride)
    
    #standardize input data
    means, devs = np.mean(arr, axis=0), np.std(arr, axis=0)
    arr_norm = standardize_data(arr, means, devs, num_sd_trim)
    approx_vals_norm = standardize_data(approx_vals, means, devs, num_sd_trim)
    
    #generate histogram from training data and blur histogram
    H, edges = np.histogramdd(arr_norm, bins = [bin_edges for _ in range(k)], density=True)
    kernel = np.ones(tuple([3 for _ in range(k)]))
    kernel = kernel / (3**k)
    convolved_H = convolve(H, kernel, mode='same')
    
    offset = 0.000001
    idx_approx_vals = ((approx_vals_norm + num_sd_trim) / stride - offset).astype(int)
    
    idx_transf = tuple([idx_approx_vals[:,i] for i in range(k)])
    probs = convolved_H[idx_transf]
    probs[probs < 0] += offset
    
    return probs, convolved_H



