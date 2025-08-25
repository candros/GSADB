## author: luo xin, creat: 2021.6.18, modify: 2021.7.14
## Modified by Charles Andros to use rasterio instead of gdal (2025.3.24)

import numpy as np
import rasterio

### tiff image reading
def readTiff(path_in):
    '''
    return: 
        img: numpy array, exent: tuple, (x_min, x_max, y_min, y_max) 
        proj info, and dimentions: (row, col, band)
    '''
    with rasterio.open(path_in, 'r') as src:
        img_array = src.read().astype(np.float64)
        meta = src.meta
        
    im_col = meta['width']
    im_row = meta['height']
    im_bands = meta['count']
    im_geotrans = meta['transform']  
    left = im_geotrans[2]
    up = im_geotrans[5]
    right = left + im_geotrans[0] * im_col + im_geotrans[1] * im_row
    bottom = up + im_geotrans[4] * im_row + im_geotrans[3] * im_col
    extent = (left, right, bottom, up)
    espg_code = str(meta['crs'].to_epsg())
    
    img_info = {'geoextent': extent,
                'geotrans':im_geotrans,
                'geosrs': espg_code,
                'row': im_row, 
                'col': im_col,
                'bands': im_bands}

    if im_bands > 1:
        img_array = np.transpose(img_array, (1, 2, 0)) # 
        return img_array, img_info 
    else:
        return img_array, img_info

###  .tiff image write
def writeTiff(im_data, im_geotrans, im_geosrs, path_out):
    '''
    input:
        im_data: tow dimentions (order: row, col),or three dimentions (order: row, col, band)
        im_geosrs: espg code correspond to image spatial reference system.
    '''
    im_data = np.squeeze(im_data)
    if 'int8' in im_data.dtype.name:
        datatype = 'uint8'
    elif 'int16' in im_data.dtype.name:
        datatype = 'uint16'
    else:
        datatype = 'float64'
        
    if len(im_data.shape) == 3:
        im_data = np.transpose(im_data, (2, 0, 1))
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands,(im_height, im_width) = 1,im_data.shape
    
    crs = rasterio.crs.CRS({"init": "epsg:"+im_geosrs})
    
    out_meta = {'driver': 'GTiff',
               'dtype': datatype,
               'nodata': None,
               'crs': crs,
               'width': im_width,
               'height': im_height,
               'count': im_bands,
               'transform': im_geotrans
               }
        
    with rasterio.open(path_out, 'w', **out_meta) as dest:
        dest.write(im_data)