
from osgeo import gdal, ogr, osr

def raster_to_shp(vector, filename):
    """
    This function is a work-in-progress. Not to be used. 
    """
    
    src_ds = gdal.Open(vector)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    srcband = src_ds.GetRasterBand(1)
    
    dst_layername = filename[:-4]
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(filename)
    
    dst_layer = dst_ds.CreateLayer(dst_layername, srs = srs)
    
    newField = ogr.FieldDefn('polygon', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    
    gdal.Polygonize(srcband, None, dst_layer, 0, [], callback=None )
    
    return True
    

def array2D_to_geoJson(geoJsonFileName, array2d,
                      layerName="BuildingID",fieldName="BuildingID"):
    """
    This function is a work-in-progress. Not to be used. 
    """
    
    memdrv = gdal.GetDriverByName('MEM')
    src_ds = memdrv.Create('', array2d.shape[1], array2d.shape[0], 1)
    band = src_ds.GetRasterBand(1)
    band.WriteArray(array2d)

    drv = ogr.GetDriverByName("geojson")
    dst_ds = drv.CreateDataSource(geoJsonFileName)
    dst_layer = dst_ds.CreateLayer(layerName, srs=None)

    fd = ogr.FieldDefn(fieldName, ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = 0

    gdal.Polygonize(band, None, dst_layer, dst_field, [], callback=None)
    
    return True