from osgeo import gdal, osr
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import convolve
import pickle

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

fpaths = {}

#NASADEM - Elevation
fpaths['cgsm_elevation'] = 'C://Users/ritvik/Desktop/JPLProject/data/NASADEMHeights/cgsm_elevation_lim.tif'

#SENTINEL - Optical Satellite
fpaths['cgsm_sentinel_infrared'] = 'C:/Users/ritvik/Desktop/JPLProject/data/mosaics/S2GM_M10_20200601_20200630_cgsm_10m_2020_July_STD_v1.3.0_97888/cgsm_10m_2020_July/corrected_B08_M10_20200601_cgsm_10m_2020_July.tiff'

#SENTINEL - Clouds
fpaths['cgsm_sentinel_clouds'] = 'C:/Users/ritvik/Desktop/JPLProject/data/mosaics/S2GM_M10_20200601_20200630_cgsm_10m_2020_July_STD_v1.3.0_97888/cgsm_10m_2020_July/quality_cloud_confidence_M10_20200601_cgsm_10m_2020_July.tiff'

#CIFOR - Baseline Wetland
fpaths['cgsm_wetlands_cifor'] = 'C://Users/ritvik/Desktop/JPLProject/data/CIFORWetlands/cifor_wetlands_cgsm.tif'

def split(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def create_raster(output_path, columns, rows, nband=1, gdal_data_type=gdal.GDT_Int32, driver=r'GTiff'):
    ''' 
    returns gdal data source raster object 
    '''
    
    # create driver
    driver = gdal.GetDriverByName(driver)

    output_raster = driver.Create(output_path, columns, rows, nband, eType = gdal_data_type)    
    
    return output_raster

def np_array_to_raster(output_path, arr, geotransform, nband=1, gdal_data_type=gdal.GDT_Int32, spatial_reference_system_wkid=4326, driver=r'GTiff'):
    ''' 
    returns a gdal raster data source

    keyword arguments:

    output_path -- full path to the raster to be written to disk
    numpy_array -- numpy array containing data to write to raster
    upper_left_tuple -- the upper left point of the numpy array (should be a tuple structured as (x, y))
    cell_resolution -- the cell resolution of the output raster
    nband -- the band to write to in the output raster
    gdal_data_type -- gdal data type of raster (see gdal documentation for list of values)
    spatial_reference_system_wkid -- well known id (wkid) of the spatial reference of the data
    driver -- string value of the gdal driver to use
    '''

    rows, columns = arr.shape

    # create output raster
    output_raster = create_raster(output_path, columns, rows, nband, gdal_data_type) 

    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(spatial_reference_system_wkid)
    output_raster.SetProjection(spatial_reference.ExportToWkt())
    output_raster.SetGeoTransform(geotransform)
    
    output_band = output_raster.GetRasterBand(1)
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

def trim_by_num_sd(arr, num_sd_trim):
    """
    arr: input array
    num_sd_trim: the number of standard deviations outside of which to remove values in arr
    """
    
    mu = np.nanmean(arr)
    dev = np.nanstd(arr)
    return arr[(arr < mu + num_sd_trim*dev)&(arr > mu - num_sd_trim*dev)]

def get_approx_densities(arr, min_val, max_val, num_sections, approx_vals):
    """
    arr: input array
    num_sections: the number of sections to break the distribution of arr into
    approx_vals: the values whose densities to approximate 
    """
    
    stride = (max_val - min_val) / num_sections
    intervals = np.arange(min_val, max_val, stride)
    
    frac_before_min = np.mean(arr < min_val)
    frac_after_max = np.mean(arr > max_val)
    
    approx_densities = []
    for i in intervals:
        approx_densities.append(np.mean((arr>i-stride/2)&(arr<i+stride/2)))
    
    approx_densities[0] = frac_before_min + .0001
    approx_densities[-1] = frac_after_max + .0001
    
    approx_densities = np.array(approx_densities)
    
    idxs = ((approx_vals - min_val) / stride).astype(int)
    idxs[idxs > num_sections-1] = num_sections-1
    idxs[idxs < 0] = 0
    ad = approx_densities[idxs]
    
    return ad

def get_interior_wetland_distribution(baseline_arr, split_band_arr, tgt_val, band_no_data_value, kernel_size=3, thresh_learning=0.1, num_sections=100, num_sd_trim=None, band_name=None, pickle_file_name=None):
    """
    baseline_arr: a baseline image of wetland locations
    split_band_arr: the band_arr split into pieces
    tgt_val: the numeric value in the baseline_arr designating wetlands
    kernel_size: the size of the high pass filter to apply to the baseline to find edges
    thresh: the boundary value when deciding between interior points and edges
    num_sd_trim: the number of standard deviations beyond which to trim the band_arr values
    """
    
    #let the edge detection threshold be half of the learning threshold
    thresh_edge_detection = thresh_learning / 2
    
    #apply a filter to the baseline wetlands image to find edge regions
    kernel = -np.ones((kernel_size,kernel_size)) / (kernel_size**2 - 1)
    kernel[kernel_size//2,kernel_size//2] = 1

    specific_wetlands = baseline_arr.copy()
    specific_wetlands[baseline_arr == tgt_val] = 1
    specific_wetlands[baseline_arr != tgt_val] = 0
    
    vicinity_score = abs(apply_convolution(specific_wetlands, kernel))
    
    #interior points are wetlands confidently below the threshold
    interior_condition = (specific_wetlands==1)&(vicinity_score < thresh_learning)
    interiors = np.where(interior_condition == 1)
    interior_vicinity_scores = vicinity_score[interiors]
    
    #edge points are above the threshold
    edges_condition = (vicinity_score >= thresh_edge_detection)
    edges = np.where(edges_condition == 1)
    
    #exterior points are non-wetlands confidently below the threshold
    exterior_condition = (specific_wetlands==0)&(vicinity_score < thresh_learning)
    exteriors = np.where(exterior_condition == 1)
    exterior_vicinity_scores = vicinity_score[exteriors]
    
    #if interiors or edges or exteriors turns up empty, return empty arrays
    if len(interiors[0]) == 0 or len(edges[0]) == 0 or len(exteriors[0]) == 0:
        return None
    
    #get the values in the SAR image lining up with interior points, sample according to confidence
    chosen_interiors = sample_by_vicinity_scores(interiors, interior_vicinity_scores, thresh_learning)
    interior_band_vals = split_band_arr[chosen_interiors].flatten()
    
    #get the values in the SAR image lining up with exterior points, sample according to confidence
    chosen_exteriors = sample_by_vicinity_scores(exteriors, exterior_vicinity_scores, thresh_learning)
    exterior_band_vals = split_band_arr[chosen_exteriors].flatten()
    
    #remove any no data values
    interior_band_vals = interior_band_vals[interior_band_vals != band_no_data_value]
    exterior_band_vals = exterior_band_vals[exterior_band_vals != band_no_data_value]
    
    #trim these values if needed
    if num_sd_trim != None:
        interior_band_vals = trim_by_num_sd(interior_band_vals, num_sd_trim)
        exterior_band_vals = trim_by_num_sd(exterior_band_vals, num_sd_trim)
      
    if band_name != None:
        plt.figure(figsize=(10,8))
        
        plt.subplot(2,1,1)
        plt.hist(interior_band_vals, density=True, edgecolor='k')
        mu = np.nanmean(interior_band_vals)
        median = np.nanmedian(interior_band_vals)
        plt.axvline(mu, color='k')
        plt.axvline(median, color='r', linestyle='--')
        plt.ylabel('Density', fontsize=16)
        plt.xlabel(band_name, fontsize=16)
        plt.title('Wetland', fontsize=18)
        
        plt.subplot(2,1,2)
        plt.hist(exterior_band_vals, density=True, edgecolor='k')
        mu = np.nanmean(exterior_band_vals)
        median = np.nanmedian(exterior_band_vals)
        plt.axvline(mu, color='k')
        plt.axvline(median, color='r', linestyle='--')
        plt.ylabel('Density', fontsize=16)
        plt.xlabel(band_name, fontsize=16)
        plt.title('Non-Wetland', fontsize=18)
        
        plt.tight_layout()
        plt.savefig('high_conf_values_%s.png'%band_name)
        
    #store the interior and exterior band distributions in pickle files for the app
    pickle.dump(interior_band_vals, open("%s_interior.p"%pickle_file_name, "wb"))
    pickle.dump(exterior_band_vals, open("%s_exterior.p"%pickle_file_name, "wb"))
      
    #get only the chunks in the SAR band chunks matching up to edges
    split_band_arr_edges = split_band_arr[edges]
    
    #we will assign continuous scores to the upsampled wetland edge values
    min_val = max(interior_band_vals.min(), exterior_band_vals.min())
    max_val = min(interior_band_vals.max(), exterior_band_vals.max())

    approx_densities_interior = get_approx_densities(interior_band_vals, min_val, max_val, num_sections, split_band_arr_edges)
    approx_densities_exterior = get_approx_densities(exterior_band_vals, min_val, max_val, num_sections, split_band_arr_edges)
    
    scores = approx_densities_interior / approx_densities_exterior
    
    #normalize scores to 0-1
    min_score = scores.min()
    max_score = scores.max()
    scores = (scores - min_score)/(max_score-min_score)
    
    return interiors, edges, scores

def upsample(orig_arr, scale, interiors, edges, var_fills, rescale_factor=1, post_upsample_blur=None):
    """
    orig_arr: the original, low resultion image
    scale: the degree of upsampling
    interiors: the coordinates of orig_arr to mark as definite wetland
    edges: the coordinates of orig_arr to give more continuous wetland confidence
    var_fills: the continuous weland confidence scores
    rescale_factor: by default, the result will have float values in 0-1, this changes that to ints in 0-rescale_factor
    post_upsample_blur: how much to blur the resulting upsampled image to induce continuity
    """
    
    #create an empty array like the original array
    small_arr = np.zeros_like(orig_arr).astype(float)
    
    #set the interior points as definite weltands
    small_arr[interiors] = 1
    
    #create the eventual upsampled array
    upsamp_arr = np.repeat(np.repeat(small_arr, scale, axis=1), scale, axis=0)
    
    #transform indices for the edges
    scaled_edge_x = np.repeat(edges[0]*scale, scale**2) + np.tile(np.repeat(np.arange(scale), scale), len(edges[0]))
    scaled_edge_y = np.repeat(edges[1]*scale, scale**2) + np.tile(np.tile(np.arange(scale), scale), len(edges[1]))
    scaled_edges = (scaled_edge_x, scaled_edge_y)
    
    #vectorized setting of all continious edge wetland confidences to the appropriate portions of the upsampled array
    upsamp_arr[scaled_edges] = var_fills.flatten()
    
    #rescale upsampled array
    upsamp_arr = (upsamp_arr*rescale_factor).astype(int)
    
    #blur the upsampled array if needed
    if post_upsample_blur != None:
        post_upsample_blur_kernel = np.ones((post_upsample_blur,post_upsample_blur)) / (post_upsample_blur**2)
        upsamp_arr = apply_convolution(upsamp_arr, post_upsample_blur_kernel)
    
    return upsamp_arr

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
        height_fine, width_fine = fine_res_arr.shape
        
        height_fine = int_scale * (height_fine // int_scale)
        width_fine = int_scale * (width_fine // int_scale)
        
        #save new fine dataset
        options_fine = gdal.WarpOptions(format='GTiff', width=width_fine, height=height_fine)
        ds = gdal.Warp(destNameOrDestDS='rescaled_fine.tiff', srcDSOrSrcDSTab=fine_res_ds, options=options_fine)
        ds = None
        
        height_coarse = height_fine // int_scale
        width_coarse = width_fine // int_scale
        
        #save new coarse datset
        options_corase = gdal.WarpOptions(format='GTiff', width=width_coarse, height=height_coarse)
        ds = gdal.Warp(destNameOrDestDS='rescaled_coarse.tiff', srcDSOrSrcDSTab=coarse_res_ds, options=options_corase)
        ds = None
            
    return int_scale, remainder

def apply_high_pass_filter(img):
    """
    img: the input image
    """
    
    #apply a horizontal edge detector
    h_sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    diff_arr_h = abs(apply_convolution(img, h_sobel)) / 4
    
    #apply a vertical edge detector
    v_sobel = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    diff_arr_v = abs(apply_convolution(img, v_sobel)) / 4
    
    #euclidean norm of the horizontal and vertical
    diff_arr_both = np.sqrt(diff_arr_h**2 + diff_arr_v**2)
    
    return diff_arr_both

def create_upsampled_wetlands_map(metric, change, tgt_type, no_data_value, high_pass_filter_size, thresh_learning, output_path_upsamp, pickle_file_name):
    """
    metric: the metric to use for distribution-based upsampling
    change: T/F. True if we should take a spatial gradient of the metric as our input, otherwise False
    tgt_type: the type of wetland we are trying to upsample
    no_data_value: the numeric designation for NO_DATA for the metric chosen
    high_pass_filter_size: the size of the wetland cell edge detection kernel
    thresh_learning: 0-1 edges threshold; closer to 0 means more edges and less interior/exterior
    output_path_upsamp: where to store the final upsampled TIFF
    pickle_file_name: the name of the pickle file to store the interior and exterior distributions
    """

    #get source dataset and wetlands dataset
    print('-------------------------------------------------')
    print('Fetching Wetlands Dataset and Source Dataset...')
    print('-------------------------------------------------')
    source_name = 'cgsm_%s'%metric
    ds_source = gdal.Open(fpaths[source_name], gdal.GA_ReadOnly)
    ds_wetlands = gdal.Open(fpaths['cgsm_wetlands_cifor'], gdal.GA_ReadOnly)
    
    #get the scale of the upsampling, saving the resized datasets
    print('-------------------------------------------------')
    print('Calculating Upscaling Factor...')
    print('-------------------------------------------------')
    scale, _ = get_scale(ds_source, ds_wetlands, True)

    #grab the resized datasets
    ds_source = gdal.Open('rescaled_fine.tiff', gdal.GA_ReadOnly)
    ds_wetlands = gdal.Open('rescaled_coarse.tiff', gdal.GA_ReadOnly)
    
    #read the arrays from the datasets
    print('-------------------------------------------------')
    print('Reading Datasets to Arrays...')
    print('-------------------------------------------------')
    wetlands_arr = ds_wetlands.ReadAsArray()
    source_arr = ds_source.ReadAsArray()
    
    #run a H and V sobel and combine to get edges
    if change:
        print('-------------------------------------------------')
        print('Calculating Spatial Gradient...')
        print('-------------------------------------------------')
        source_arr = apply_high_pass_filter(source_arr)
    
    #split the finer resolution array into small squares
    print('-------------------------------------------------')
    print('Spliting Finer Resolution Array...')
    print('-------------------------------------------------')
    baseline_h, baseline_w = wetlands_arr.shape
    band_h, band_w = source_arr.shape
    split_band_arr = split(source_arr, scale, scale).reshape(baseline_h, baseline_w, scale, scale)
    
    #get target value
    tgt_val = feature_to_code[tgt_type]
    
    #get sure wetland elevations, unsure wetland edges, and scores to set into edges
    print('-------------------------------------------------')
    print('Getting Interior / Exterior / Edge Values...')
    print('-------------------------------------------------')
    interiors, edges, scores = get_interior_wetland_distribution(wetlands_arr, split_band_arr, tgt_val, no_data_value, high_pass_filter_size, thresh_learning, 10, 2, 'test', pickle_file_name)
    
    #upsample the baseline
    print('-------------------------------------------------')
    print('Upsampling...')
    print('-------------------------------------------------')
    upsampled_arr = upsample(wetlands_arr, scale, interiors, edges, scores, 255, int(scale*3/2))
    
    print('-------------------------------------------------')
    print('Saving Files...')
    print('-------------------------------------------------')
    #save the upsampled map
    ds = np_array_to_raster(output_path_upsamp, upsampled_arr, ds_source.GetGeoTransform())
    ds = None

    #save the original baseline map
    formatted_feature = tgt_type.replace('/','').replace(' ','')
    output_path_orig = 'assets/%s_original.tiff'%formatted_feature
    ds = np_array_to_raster(output_path_orig, wetlands_arr == tgt_val, ds_wetlands.GetGeoTransform())
    ds = None
    
    #save the diff of the source arr
    if change:
        ds = np_array_to_raster('assets/%s_diff.tiff'%metric, source_arr, ds_source.GetGeoTransform())
        ds = None
    
    