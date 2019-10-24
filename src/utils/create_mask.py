
import buzzard as buzz

aoi_list = ["aoi1", "aoi2"]

def create_mask(rgb_path, shp_path, mask_path):

    ds = buzz.DataSource(allow_interpolation=True)
    ds.open_raster('rgb', rgb_path)
    ds.open_vector('shp', shp_path)

    fp= buzz.Footprint(
        tl=ds.rgb.fp.tl,
        size=ds.rgb.fp.size,
        rsize=ds.rgb.fp.rsize,)

    polygons = ds.shp.iter_data(None)

    mask = fp.burn_polygons(polygons)
    mask_tr = mask *255

    with ds.create_raster('mask', mask_path, ds.rgb.fp, 'uint8', 1,
                      band_schema=None, sr=ds.rgb.proj4_virtual).close:
        ds.mask.set_data(mask_tr, band = 1)

    return True

if __name__ == "__main__":

    for aoi in aoi_list:

        images_dir = "data/datasets/"
        rgb_path = images_dir + "images/" + aoi+".tif"
        shp_path = images_dir + "shapes/" + aoi+".shp"
        mask_path = images_dir + "masks/" + aoi+".tif"

        create_mask(rgb_path, shp_path, mask_path)
