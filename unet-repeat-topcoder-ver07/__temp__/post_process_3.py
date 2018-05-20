from common import *
from utility.draw import *
from dataset.reader import *


# https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741
#    [ods.ai] topcoders, 1st place solution

# see laso:
#     https://www.kaggle.com/c/data-science-bowl-2018/discussion/54742
#     https://www.kaggle.com/c/data-science-bowl-2018/discussion/47590

#  https://spark-in.me/post/playing-with-dwt-and-ds-bowl-2018
#  https://github.com/snakers4/ds_bowl_2018/blob/master/src/utils/watershed.py

#  https://www.youtube.com/watch?v=RrJMEKohwrs
#  http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
#  http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.watershed
#  https://docs.opencv.org/3.1.0/d3/db4/tutorial_py_watershed.html



##  erode + dilate  #<todo>
def erode_dilate_post_process(mask, centers):
    #erode
    #dilate
    return mask


## watershed post processing
##    https://www.pyimagesearch.com/2015/11/02/watershed-opencv/

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

# def watershed_post_process(binary, centers ):
#
#     distance = cv2.distanceTransform(binary, cv2.DIST_L2, maskSize=5 )
#
#
#     cleaned_mask  = clean_mask(image, contour)
#     good_markers  = get_markers(cleaned_mask, contour)
#     good_distance = get_distance(cleaned_mask)
#
#     labels = morph.watershed(-distance, markers, mask)
#
#
#
#     # noise removal
#     kernel = np.ones((3,3),np.uint8)
#     opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#     # sure background area
#     sure_bg = cv2.dilate(opening,kernel,iterations=3)
#
#     # Finding sure foreground area
#
#
#     # adaptive hist for distance transform
#     dist_transform = exposure.equalize_adapthist(((dist_transform / dist_transform.max()) * 255).astype('uint8'), clip_limit=0.03)
#
#     ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#
#     # Finding unknown region
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg,sure_fg)
#
#     # Marker labelling
#     ret, markers = cv2.connectedComponents(sure_fg)
#
#     distance = ndi.distance_transform_edt(mask)
#
#     labels = watershed(-distance, markers, mask)
#
#     return mask

# check #################################################################


# http://answers.opencv.org/question/958/i-need-a-maxenclosingcircle-function/
def find_max_enclosed_circle(instance):

    distance = ndimage.distance_transform_edt(instance>0)
    radius = np.amax(distance) #index of flattened arrays
    index  = np.argmax(distance) #index of flattened arrays
    y,x    = np.unravel_index(index, distance.shape)#convert to 2D index

    #<todo>
    # check if multiple maxima exist and how many
    ##a = len(distance[dista nce==radius])

    return int(y), int(x), int(radius)



def make_maker(instance):
    area = instance.sum()
    radius = (area/math.pi)**0.5
    r = int(max(1,radius))
    instance = instance.astype(np.uint8)*255
    maker = cv2.erode(instance, np.ones((r,r), np.uint8))
    maker = maker>128
    return maker



def mask_to_marker1(mask):

    H,W = mask.shape[:2]
    num_instances = mask.max()

    maker = np.zeros((H,W),np.int32)
    for i in range(num_instances):
        instance = mask==i+1
        #y,x,radius = find_circle(instance)
        m =  make_maker(instance)
        maker[m] = i+1
        #<debug>
        if 0:
            #instance = instance.astype(np.uint8)*255
            instance[maker]=128
            ## instance = cv2.merge([instance,instance,instance])
            # cv2.circle(instance,(x,y),radius,(0,0,255),1)
            #image_show('instance',instance,resize=1)
            image_show_norm('maker',maker,resize=1)
            #image_show('color1_overlay',color1_overlay,resize=1)
            #image_show('color_overlay',color_overlay,resize=1)
            cv2.waitKey(0)

    return maker


def mask_to_marker(mask):

    H,W = mask.shape[:2]
    yy,xx= np.mgrid[:H, :W]
    num_instances = mask.max()

    maker = np.zeros((H,W),np.int32)
    for i in range(num_instances):
        instance = mask==i+1
        y,x,radius = find_max_enclosed_circle(instance)
        r = int(min(5,radius))#int(max(1,radius*0.25))

        m = (xx - x) ** 2 + (yy - y) ** 2 <r**2
        maker[m] = i+1
        #<debug>
        if 0:
            #instance = instance.astype(np.uint8)*255
            instance[maker]=128
            ## instance = cv2.merge([instance,instance,instance])
            # cv2.circle(instance,(x,y),radius,(0,0,255),1)
            #image_show('instance',instance,resize=1)
            image_show_norm('maker',maker,resize=1)
            #image_show('color1_overlay',color1_overlay,resize=1)
            #image_show('color_overlay',color_overlay,resize=1)
            cv2.waitKey(0)

    return maker
# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    folder='stage1_test'
    #name ='697a05c6fe4a07c601d46da80885645ad574ea19b47ee795ccff216c9f1f1808'
    name ='699f2992cd71e2e28cf45f81347ff22e76b37541ce88087742884cd0e9aadc68'
    #name ='912a679e4b9b1d1a75170254fd675b8c24b664d80ad7ea7e460241a23535a406'
    #name ='1747f62148a919c8feb6d607faeebdf504b5e2ad42b6b1710b1189c37ebcdb2c'

    image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png'%(folder,name), cv2.IMREAD_COLOR)
    mask  = np.load( DATA_DIR + '/image/%s/masks/%s.npy'%(folder,name)).astype(np.int32)

    H,W = mask.shape[:2]
    distance = np.zeros((H,W),np.float32)
    cut      = np.zeros((H,W),np.float32)
    binary   = (mask!=0)
    num_instances = mask.max()

    border = np.zeros((H,W),np.float32)
    cut = np.zeros((H,W),np.float32)
    cum_count = np.zeros((H,W),np.float32)
    for i in range(num_instances):
        instance = mask==i+1
        d = ndimage.distance_transform_edt(instance)
        d_max = d.max()
        d = d/(d_max+0.01)

        e = ndimage.distance_transform_edt(~instance)
        e = np.clip(0.8-e/(d_max+0.01),0,0.8)

        border = border+1-d
        cut = cut+e
        cum_count = cum_count + (e>0)


        image_show_norm('border',border,resize=1)
        image_show_norm('cut',cut,resize=1)
        image_show_norm('cum_count',cum_count,resize=1)
        cv2.waitKey(1)


    cut[cut<1.3]=0
    f = binary*cut

    contour_overlay = cv2.cvtColor(((
        f/f.max()   #cum_count/cum_count.max()
    )*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    contour_overlay = mask_to_contour_overlay(mask,contour_overlay)

    #image_show_norm('cut',cut/(cum_count+0.0001),resize=1)
    image_show('contour_overlay',contour_overlay,resize=1)
    cv2.waitKey(0)
    exit(0)


    distance = distance+d
    for j in range(num_instances):
        if j==i: continue
        cut +=  np.clip(1-outer_borders[j]/(d_max+0.01),0,1)


    f = cut*d





    image_show_norm('d',d,resize=1)
    #image_show_norm('e',e,resize=1)
    image_show_norm('f',f,resize=1)
    image_show_norm('cut',cut,resize=1)
    #image_show_norm('inv_instance',inv_instance,resize=1)
    image_show('contour_overlay',contour_overlay,resize=1)
    cv2.waitKey(0)

    # distance = distance.astype(np.float32)
    #
    #
    #
    #
    # image_show_norm('binary',binary,resize=1)
    # image_show_norm('distance',distance,resize=1)
    # image_show_norm('marker',marker,resize=1)
    # image_show_norm('water',water,resize=1)
    # image_show('color1_overlay',color1_overlay,resize=1)
    # #image_show('color_overlay',color_overlay,resize=1)
    # cv2.waitKey(0)
    #
    # peaks    = peak_local_max(distance, binary, footprint=np.ones((3, 3)), indices=False)
    # markers  = ndi.label(peaks)[0]
    #
    #
    #
    # watershed_post_process(mask)