import numpy as np 
from math import ceil 
from math import floor

# input -> a np.array and target size (i.e. 254)
# return a list of padded tiles (np.arrays) and 2 indexes useful for untiling

def tile(im_array,target_size):
    im_size = im_array.shape
    height = im_size[0]
    width = im_size[1]
    tile_shape = [target_size,target_size,3]
    
    y = []
    x = []
    
    for i in range(0,width-ceil(target_size/2),ceil(target_size/2)):
        y.append(i)
    for i in range(0,height-ceil(target_size/2),ceil(target_size/2)):
        x.append(i)
    
    tiles = []

    for i in x:
        for j in y:
            temp = np.array(np.zeros(tile_shape))
            temp[i:i+target_size,j:j+target_size,:] = im_array[i:i+target_size,j:j+target_size,:]
            tiles.append(temp)
    
    return tiles

# input padded tiles and original shape [h,w,3]
# tile size an integer
def untile(tiles,tile_size,original_shape):
    target = np.array(np.zeros(original_shape))
    
    height = original_shape[0]
    width = original_shape[1]

    x = (floor(width/(tile_size/2))-1)*(tile_size/2)
    y = (floor(height/(tile_size/2))-1)*(tile_size/2)

    x_l = x + tile_size
    y_l = y + tile_size
    array = np.array(np.zeros([x_l,y_l,3]))

    count = 0

    for i in x:
        for j in y:
            array[i:(i+tile_size),j:(j+tile_size),:] = np.uint8(tiles[count])
            count += 1
    
    target = array[:original_shape[0],:original_shape[1],:]
    return target

