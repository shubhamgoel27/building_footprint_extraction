import numpy as np
import itertools
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def img_to_slices(img, img_rows, img_cols):
    """
    Convert image into slices
        args:
            .img: input image
            .img_rows: img_rows of slice
            .img_cols: img_rows of slice
        return slices, shapes[nb_rows, nb_cols]
    """
    nb_rows = img.shape[0] // img_rows
    nb_cols = img.shape[1] // img_cols
    slices = []
    
    # generate img slices
    for i, j in itertools.product(range(nb_rows), range(nb_cols)):
        slice = img[i * img_rows: i * img_rows + img_rows,
                    j * img_cols:j * img_cols + img_cols]
        slices.append(slice)
        
    return slices, [nb_rows, nb_cols]


def slices_to_img(slices, shapes):
    """
    Restore slice into image
        args:
            slices: image slices
            shapes: [nb_rows, nb_cols] of original image
        return img
    """
    # set img placeholder
    if len(slices[0].shape) == 3:
        img_rows, img_cols, in_ch = slices[0].shape
        img = np.zeros(
            (img_rows * shapes[0], img_cols * shapes[1], in_ch), np.uint8)
    else:
        img_rows, img_cols = slices[0].shape
        img = np.zeros((img_rows * shapes[0], img_cols * shapes[1]), np.uint8)
        
    # merge
    for i, j in itertools.product(range(shapes[0]), range(shapes[1])):
        img[i * img_rows:i * img_rows + img_rows,
            j * img_cols:j * img_cols + img_cols] = slices[i * shapes[1] + j]
        
    return img


def save_np(np_array, filepath):
    
    im = Image.fromarray(np_array.astype('uint8'))
    im.save(filepath)
#    matplotlib.image.imsave(filename, np_array)
    
    return True


def show_image(images, labels, preds = None, n=3, figx=10, figy =8):
    
    l = 3
    if preds is None:
        l = 2
    
    f = plt.figure(figsize=(figx, figy), dpi= 80, facecolor='w', edgecolor='k')
    f.add_subplot(1,l, 1)
    plt.imshow(images[n,:,:,:])
    
    f.add_subplot(1,l, 2)
    plt.imshow(labels[n,:,:,0], cmap="gray")
    
    if l==3:
        f.add_subplot(1,3, 3)
        plt.imshow(preds[n,:,:,0], cmap="gray")


def downsample():
    ds = 2
    pilimg = Image.open('data/test/rcp2/rcp2.tiff')
    #pilimg.show()

    h, w = pilimg.size
    pilimg_resized = pilimg.resize((h//ds, w//ds),PIL.Image.LANCZOS)
    pilimg_resized.save('data/test/rcp2/rcp2_75.tiff')
    
    return True
    
def plot_hist(image, mode='rgb'):
    
    if mode=='rgb':
        color = ('r','g','b')
    else:
        color = ('b','g','r')
        
    for i,col in enumerate(color):
        histr = cv2.calcHist([image], [i], None, [256], [0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show() 
    
    