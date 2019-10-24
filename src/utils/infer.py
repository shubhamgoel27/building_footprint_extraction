
import buzzard as buzz
import numpy as np

import tqdm
from os.path import isfile, join
from os import listdir

def tile_image(tile_size, ds, fp):
    '''
    Tiles image in several tiles of size (tile_size, tile_size)
    Params
    ------
    tile_siz : int
        size of a tile in pixel (tiles are square)
    ds: Datasource
        Datasource of the input image (binary)
    fp : footprint
        global footprint 
    Returns
    -------
    rgb_array : np.ndarray
        array of dimension 5 (x number of tiles, y number of tiles,
        x size of tile, y size of tile, number of canal)
    tiles : np.ndarray of footprint
        array of of size (x number of tiles, y number of tiles) 
        that contains footprint information
    
        
    '''
    tiles = fp.tile((tile_size,tile_size), overlapx=0, overlapy=0,
             boundary_effect='extend', boundary_effect_locus='br')
    
    rgb_array = np.zeros([tiles.shape[0],tiles.shape[1],tile_size,tile_size,3],
                         dtype='uint8')
        
    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            rgb_array[i,j] = ds.rgb.get_data(band=(1,2,3), fp=tiles[i,j])
            
    return rgb_array, tiles

def untile_and_predict(rgb_array,tiles,fp,model, pre_process):
    '''
    Get the tile binary array and the footprint matrix and returns the reconstructed image
    Params
    ------
    rgb_array : np.ndarray
        array of dimension 5 (x number of tiles, y number of tiles,
        x size of tile, y size of tile,3)
    tiles : np.ndarray of footprint
        array of of size (x number of tiles, y number of tiles) 
        that contains footprint information
    Returns
    -------
    rgb_reconstruct : np.ndarray
        reconstructed rgb image
    '''
    # initialization of the reconstructed rgb array
    binary_reconstruct = np.zeros([
        rgb_array.shape[0]*rgb_array.shape[2], #pixels along x axis
        rgb_array.shape[1]*rgb_array.shape[3], # pixels along y axis
        ], dtype='float32')
    
    for i in tqdm.tqdm(range(tiles.shape[0])):
        for j in range(tiles.shape[1]):
            tile_size = tiles[i,j].rsize
            
            # predict the binaryzed image
            image = rgb_array[i,j]
            image = pre_process(image)
            predicted = predict_image(image,model)
            
            # add the image in the global image
            binary_reconstruct[i*tile_size[0]:(i+1)*tile_size[0],
                            j*tile_size[1]:(j+1)*tile_size[1]
                            ] = predicted
            
    # delete the tilling padding
    binary_reconstruct = binary_reconstruct[:fp.rsize[1],:fp.rsize[0]]
    
    return binary_reconstruct

def predict_image(image,model):
    '''
    Predict one image with the model. Returns a binary array
    Parameters
    ----------
    image : np.ndarray
        rgb input array of the (n,d,3)
    model : Model object
        trained model for the prediction of image (tile_size * tile_size)
    '''
    shape_im = image.shape
    
    predicted_image = model.predict(image.reshape(1,shape_im[0], shape_im[1],3))    
    predicted_image = predicted_image.reshape(shape_im[0],shape_im[1])
    
    return predicted_image

def predict_map(model,tile_size,ds_rgb,fp, pre_process):
    '''
    Pipeline from the whole rasper and footprint adapted to binary array
    Params
    ------
    model : Model object
        trained model for the prediction of image (tile_size * tile_size)
    tile_size : int
        size of tiles (i.e. size of the input array for the model)
    ds_rgb : datasource
        Datasource object for the rgb image
    fp : footprint
        footprint of the adapted image (with downsampling factor)
    '''
    print("Tiling images...")
    rgb_array, tiles = tile_image(tile_size, ds_rgb, fp)
    
    print("Predicting tiles..")
    predicted_binary = untile_and_predict(rgb_array,tiles,fp,model, pre_process)
    
    return predicted_binary

def predict_from_file(rgb_path, model, pre_process, downsampling_factor=3,tile_size=128):
    '''
    Predict binaryzed array and adapted footprint from a file_name
    Parameters
    ----------
    rgb_path : string
        file name (with extension)
    model : Model object
        trained model for the prediction of image (tile_size * tile_size)
    downsampling_factor : int
        downsampling factor (to lower resolution)
    tile_size : int
        size of a tile (in pixel) i.e. size of the input images 
        for the neural network
    '''
    ds_rgb = buzz.DataSource(allow_interpolation=True)
    ds_rgb.open_raster('rgb', rgb_path)
    
    fp= buzz.Footprint(
            tl=ds_rgb.rgb.fp.tl,
            size=ds_rgb.rgb.fp.size,
            rsize=ds_rgb.rgb.fp.rsize/downsampling_factor,
    ) #unsampling

    predicted_binary = predict_map(model, tile_size, ds_rgb, fp, pre_process)
    
    return predicted_binary, fp
    
    
def save_polynoms(file,binary,fp):
    '''
    Find polynoms in a binary array and save them at the geojson format
    Parameters
    ----------
    file : string
        file name (with extension)
    binary : np.ndarray
        array that contains the whole binaryzed image
    fp : footprint
        footprint linked to the binary array
    '''
    poly = fp.find_polygons(binary)
    
    path = 'geoJSON/'+file.split('.')[0]+'.geojson'
    ds = buzz.DataSource(allow_interpolation=True)
    ds.create_vector('dst', path, 'polygon', driver='GeoJSON')
    for p in poly:
        ds.dst.insert_data(p)
    ds.dst.close()    
    
def compute_files(model_nn,images_train,downsampling_factor,tile_size):
    '''
    Compute polynoms for each files in images_train folder and save them in
    geojson format.
    Parameters
    ----------
    model : Model object
        trained model for the prediction of image (tile_size * tile_size)
    images_traian : string
        folder name for whole input images
    downsampling_factor : int
        downsampling factor (to lower resolution)
    tile_size : int
        size of a tile (in pixel) i.e. size of the input images 
        for the neural network
    '''
    files = [f for f in listdir(images_train) if isfile(join(images_train, f))]
    for i in range(len(files)):
        print("Processing file number "+str(i)+"/"+str(len(files))+"...")
        predicted_binary, fp = predict_from_file(files[i],model_nn,
                                        images_train,
                                        downsampling_factor,tile_size)
        save_polynoms(files[i],predicted_binary,fp)
    
    
    
def poly_to_binary(gt_path, geojson_path, downsampling_factor):
    
    ds = buzz.DataSource(allow_interpolation=True)
    ds.open_raster('binary', gt_path) 
    ds.open_vector('polygons',geojson_path, driver='geoJSON')
    
    fp = buzz.Footprint(
            tl=ds.binary.fp.tl,
            size=ds.binary.fp.size,
            rsize=ds.binary.fp.rsize/downsampling_factor ,
        )
    
    binary = ds.binary.get_data(band=(1), fp=fp).astype('uint8')
    binary_predict = np.zeros_like(binary)
    
    for poly in ds.polygons.iter_data(None):
        mark_poly = fp.burn_polygons(poly)
        binary_predict[mark_poly] = 1
        
    return binary,binary_predict
    
