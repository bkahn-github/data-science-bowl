from common import *

from dataset.transform import *
from dataset.sampler import *
from utility.file import *
from utility.draw import *



#data reader  ----------------------------------------------------------------
class ScienceDataset(Dataset):

    def __init__(self, split, transform=None, mode='train'):
        super(ScienceDataset, self).__init__()

        self.split     = split
        self.transform = transform
        self.mode      = mode

        #read split
        ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')
        self.ids = ids

        #print
        print('\tnum_ids = %d'%(len(self.ids)))
        print('')


    def __getitem__(self, index):
        id = self.ids[index]
        folder, name   = id.split('/')
        image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png'%(folder,name), cv2.IMREAD_COLOR)

        if self.mode in ['train']:
            mask = np.load( DATA_DIR + '/image/%s/masks/%s.npy'%(folder,name)).astype(np.int32)
            if self.transform is not None:
                return self.transform(image, mask, index)
            else:
                return image, mask, index

        if self.mode in ['test']:
            if self.transform is not None:
                return self.transform(image, index)
            else:
                return image, index

    def __len__(self):
        return len(self.ids)







# draw  ----------------------------------------------------------------

def color_overlay_to_mask(image):
    H,W = image.shape[:2]

    mask = np.zeros((H,W),np.int32)
    unique_color = set( tuple(v) for m in image for v in m )

    #print(len(unique_color))
    count=0
    for color in unique_color:
        #print(color)
        if color ==(0,0,0): continue

        thresh = (image==color).all(axis=2)
        label  = skimage.morphology.label(thresh)

        index = [label!=0]
        count = mask.max()
        mask[index] =  label[index]+count

    return mask


def mask_to_color_overlay(mask, image=None, color=None):

    height,width = mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()

    num_instances = int(mask.max())
    if num_instances==0: return overlay

    if type(color) in [str] or color is None:
        #https://matplotlib.org/xkcd/examples/color/colormaps_reference.html

        if color is None: color='summer'  #'cool' #'brg'
        color = plt.get_cmap(color)(np.arange(0,1,1/num_instances))
        color = np.array(color[:,:3])*255
        color = np.fliplr(color)
        #np.random.shuffle(color)

    elif type(color) in [list,tuple]:
        color = [ color for i in range(num_instances) ]

    for i in range(num_instances):
        overlay[mask==i+1]=color[i]

    return overlay



def mask_to_contour_overlay(mask, image=None, color=[255,255,255]):

    height,width = mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()

    num_instances = int(mask.max())
    if num_instances==0: return overlay

    for i in range(num_instances):
        overlay[mask_to_inner_contour(mask==i+1)]=color

    return overlay

# modifier  ----------------------------------------------------------------

def mask_to_outer_contour(mask, thickness=1):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = (~mask) & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    if thickness!=1:
        y,x = np.where(contour)
        height,width = mask.shape[:2]
        image = np.zeros((height,width), np.uint8)
        for yy,xx in zip(y,x):
            cv2.circle(image,(xx,yy),thickness,255,-1,cv2.LINE_8)

        contour = (~mask) &(image>128)
    return contour


def mask_to_inner_contour(mask, thickness=1):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )

    if thickness!=1:
        y,x = np.where(contour)
        height,width = mask.shape[:2]
        image = np.zeros((height,width), np.uint8)
        for yy,xx in zip(y,x):
            cv2.circle(image,(xx,yy),thickness,255,-1,cv2.LINE_8)

        contour = mask &(image>128)
    return contour


##-------------------------------------------------------------------------
def mask_to_distance(mask):

    H,W = mask.shape[:2]
    distance = np.zeros((H,W),np.float32)

    num_instances = mask.max()
    for i in range(num_instances):
        instance = mask==i+1
        d = ndimage.distance_transform_edt(instance)
        d = d/(d.max()+0.01)
        distance = distance+d

    distance = distance.astype(np.float32)
    return distance


def mask_to_foreground(mask):
    foreground = (mask!=0).astype(np.float32)
    return foreground


def mask_to_border(mask):
    H,W = mask.shape[:2]
    border = np.zeros((H,W),np.float32)
    distance  = mask_to_distance(mask)
    y,x = np.where( np.logical_and(distance>0, distance<0.5) )
    border[y,x] = 1
    return border



def instance_to_mask(instance):
    H,W = instance.shape[1:3]
    mask = np.zeros((H,W),np.int32)

    num_instances = len(instance)
    for i in range(num_instances):
         mask[instance[i]>0] = i+1

    return mask

def mask_to_instance(mask):
    H,W = mask.shape[:2]
    num_instances = mask.max()
    instance = np.zeros((num_instances,H,W), np.float32)
    for i in range(num_instances):
         instance[i] = mask==i+1

    return instance


def mask_to_label(mask):
    H,W = mask.shape[:2]
    label    = np.zeros((H,W),np.float32)
    distance = mask_to_distance(mask)
    label[distance>0.5]=1  #center
    label[np.logical_and(distance>0,distance<=0.5)]=2  #boundary
    return label

# def mask_to_cut(mask):
#
#     H,W    = mask.shape[:2]
#     cut    = np.zeros((H,W),np.bool)
#     inners = []
#     outers = []
#     radius = []
#
#     num_instances = mask.max()
#     for i in range(num_instances):
#         instance = mask==i+1
#         r = (instance.sum()/math.pi)**0.5
#         r = int(round(max(1,r*0.25)))
#         inners.append(mask_to_inner_contour(instance, thickness=r))
#         outers.append(mask_to_outer_contour(instance, thickness=r))
#         radius.append(r)
#
#
#     for i in range(num_instances):
#         instance = mask==i+1
#         r = radius[i]
#         c = mask_to_outer_contour(instance, thickness=r)
#
#         for j in range(num_instances):
#             if j==i : continue
#             cut |= c & inners[j]
#             cut |= c & outers[j]
#
#     return cut


# def mask_to_cut(mask):
#
#     binary = mask!=0
#     H,W    = mask.shape[:2]
#
#     cut    = np.zeros((H,W),np.bool)
#     inner  = np.zeros((H,W),np.bool)
#
#     num_instances = mask.max()
#     for i in range(num_instances):
#         instance = mask==i+1
#         radius   = (instance.sum()/math.pi)**0.5
#         r = int(round(max(1,radius*0.25)))
#         c = mask_to_inner_contour(instance, thickness=r)
#         inner |= c
#
#
#     for i in range(num_instances):
#         instance = mask==i+1
#         radius   = (instance.sum()/math.pi)**0.5
#         r = int(round(max(1,radius*0.33)))
#
#         c = mask_to_outer_contour(instance, thickness=r)
#         cut |= c & inner
#
#     return cut


# def mask_to_cut(mask):
#
#     H,W = mask.shape[:2]
#     cut = np.zeros((H,W),np.float32)
#
#     num_instances = mask.max()
#     for i in range(num_instances):
#         instance = mask==i+1
#         d = ndimage.distance_transform_edt(instance)
#         d_max = d.max()
#         d = d/(d_max+0.01)
#
#         e = ndimage.distance_transform_edt(~instance)
#         e = np.clip(0.9-e/(d_max+0.01),0,0.9)
#         cut = cut+e
#
#     cut = cut.astype(np.float32)
#     cut[cut<1.5]=0
#     cut[cut>0]=1
#     return  cut


def mask_to_cut(mask):
    H,W    = mask.shape[:2]
    dilate = np.zeros((H,W),np.bool)
    marker = np.zeros((H,W),np.bool)

    num_instances = mask.max()
    for i in range(num_instances):
        instance = mask==i+1
        d = ndimage.distance_transform_edt(instance)
        d_max = d.max()
        d = d/(d_max+1e-12)
        marker |= d>0.5

        radius = (instance.sum()/math.pi)**0.5
        r = 3 #int(round(max(1,radius*0.5)))
        dilate |= cv2.dilate(instance.astype(np.uint8),kernel=np.ones((r,r),np.float32)).astype(np.bool)

    marker   = skimage.morphology.label(marker)


    water1  = skimage.morphology.watershed(dilate, marker, mask=dilate)
    water2  = skimage.morphology.watershed(dilate, marker, mask=dilate, watershed_line=True)
    cut     = (water1-water2)!=0
    cut     = cv2.dilate(cut.astype(np.uint8),kernel=np.ones((3,3),np.float32)).astype(np.bool)

    return  cut






# check ##################################################################################3

def run_check_train_dataset_reader():

    dataset = ScienceDataset(
        'train1_ids_gray2_500',
        #'disk0_ids_dummy_9',
        #'merge1_1',
        mode='train',transform = None,
    )

    for n in range(len(dataset)):
        i=n #13  #=
        image, mask_truth, index = dataset[i]

        folder, name = dataset.ids[index].split( '/')
        print('%05d %s' %(i,name))

        # image1 = random_transform(image, u=0.5, func=process_gamma, gamma=[0.8,2.5])
        # image2 = process_gamma(image, gamma=2.5)

        #image1 = random_transform(image, u=0.5, func=do_process_custom1, gamma=[0.8,2.5],alpha=[0.7,0.9],beta=[1.0,2.0])
        #image1 = random_transform(image, u=0.5, func=do_unsharp, size=[9,19], strength=[0.2,0.4],alpha=[4,6])
        #image1 = random_transform(image, u=0.5, func=do_speckle_noise, sigma=[0.1,0.5])

        #image1, truth_mask1 = random_transform2(image, truth_mask, u=0.5, func=do_shift_scale_rotate2, dx=[0,0],dy=[0,0], scale=[1/2,2], angle=[-45,45])
        #image1, truth_mask1 = random_transform2(image, truth_mask, u=0.5, func=do_elastic_transform2, grid=[16,64], distort=[0,0.5])



        # image_show('image',image,1)
        # color_overlay = mask_to_color_overlay(truth_mask)
        # image_show('color_overlay',color_overlay,1)
        #
        #
        # image_show('image1',image1,1)
        # #image_show('image2',image2,1)
        # color_overlay1 = mask_to_color_overlay(truth_mask1)
        # image_show('color_overlay1',color_overlay1,1)


        label_truth = mask_to_annotation(mask_truth)

        contour_overlay = mask_to_contour_overlay(mask_truth,  image, [0,255,0])
        image_show('contour_overlay',contour_overlay,resize=1)


        image_show_norm('label_truth',label_truth,resize=1)
        # image_show_norm('truth_foreground',truth_foreground,resize=1)
        # image_show_norm('truth_border',truth_border,resize=1)
        cv2.waitKey(0)
        continue






# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_train_dataset_reader()

    print( 'sucess!')













