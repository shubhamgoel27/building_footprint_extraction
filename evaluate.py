
import argparse
from tqdm import tqdm

## Standard imports
import buzzard as buzz

## Custom imports
from src.utils.load_model import LoadModel
from src.utils.infer import predict_from_file
from src.utils.post_process import morphological_rescontruction, grow

def parse_args():
    """Evaluation options for INRIA building segmentation model"""
    parser = argparse.ArgumentParser(description='JioGIS \
                                     Segmentation')
    # model and dataset 
    parser.add_argument('--weights', type=str, default='data/best_inria_weights.h5',
                        help='weights file (default: data/best_inria_weights.h5)')
    parser.add_argument('--data_dir', type=str, default='../../gis/data/datasets/annotated_data/images/',
                        help='data directory (default: data/datasets/annotated_data/images/)')
    parser.add_argument('--aoi_file', type=str, default='dimapur.tif',
                        help='AoI filename (default: dimapur.tif)')
    parser.add_argument('--results_dir', type=str, default='data/test/geoJSON/',
                        help='result directory (default: data/test/geoJSON/)')
    # test hyper params
    parser.add_argument('--downsample', type=int, default=1,
                        help='downsampling factor (default:1)')
    parser.add_argument('--tile_size', type=int, default=128,
                        help='tile size for cropping the file (default: 128)')
    parser.add_argument('--threshold', type=float, default=0.5, metavar='ths',
                        help='threshold for converting probs into mask (default: 0.5)')
    #Post processing boolean
    parser.add_argument('--post_process', type=int, default=1,
                        help='post process boolean: change to 1 for post-processing the output (default:1)')
    parser.add_argument('--overlap_factor', type=int, default=1,
                        help='overlap factor for tiles (default:1)')
    # the parser
    args = parser.parse_args()
    print(args)
    return args

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.model = LoadModel.load(model_file = self.args.weights)
        
        self.rgb_path = self.args.data_dir + args.aoi_file
        self.ds_rgb = buzz.DataSource(allow_interpolation=True)
        self.ds_rgb.open_raster('rgb', self.rgb_path)
        
        self.downs = self.args.downsample
        self.tile_size = self.args.tile_size
        self.threshold = self.args.threshold
        
    def pre_process(self, image):
        image = image/127.5-1.0
        return image

    def generate_preds(self, post_process=0):
        predicted_binary, fp = predict_from_file(self.rgb_path, self.model, self.pre_process, downsampling_factor=self.downs, tile_size=self.tile_size, overlap_factor=self.args.overlap_factor)
        if post_process==1:
            print("Post-processing started...")
            morph_pred = morphological_rescontruction(predicted_binary, 0.5, 0.7)
            dilated_pred = grow(morph_pred, 3)
            predicted_binary = dilated_pred
            print("Post-processing complete...")
        return predicted_binary, fp
    
    def convert_to_polygons(self, predicted_binary, fp):
        predicted_mask = (predicted_binary > self.threshold)*255
        poly = fp.find_polygons(predicted_mask)
        ds = buzz.DataSource(allow_interpolation=True)
        self.geojson_path = args.results_dir + self.args.aoi_file[:-4] +"_ds"+str(self.downs)+str("_ths_")+str(self.threshold)+str("_inria_post_process")+'.geojson'
        ds.create_vector('dst', self.geojson_path, 'polygon', driver='GeoJSON')
        for i in tqdm(range(len(poly))):
            ds.dst.insert_data(poly[i])
        ds.dst.close()

if __name__ == "__main__":
    args = parse_args()
    print('Evaluating model on ', args.aoi_file)
    evaluator = Evaluator(args)
    predicted_binary, fp = evaluator.generate_preds(args.post_process)
    print("Converting prediction mask into geoJSON polygons...")
    evaluator.convert_to_polygons(predicted_binary, fp)
    print("File saved at ", evaluator.geojson_path)
                     