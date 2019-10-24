
import buzzard as buzz

def raster_explorer(raster_path, stats=True):
    
    ds = buzz.DataSource(allow_interpolation=True)
    ds.open_raster('rgb', raster_path)
    
    fp= buzz.Footprint(
        tl=ds.rgb.fp.tl,
        size=ds.rgb.fp.size,
        rsize=ds.rgb.fp.rsize,)
    
    tlx, dx, rx, tly, ry, dy = fp.gt
    
    if stats:
#        print("No of channels in raster image:", len(ds.rgb))
        print("Raster size:", ds.rgb.fp.rsize)
        print("Dtype of Raster:", ds.rgb.dtype)
#        print("Projection System:", ds.rgb.proj4_virtual)
#        print("Top left spatial coordinates:", (tlx, tly))
        print("Resolution in x and y:", (dx, dy))
        
    return ds