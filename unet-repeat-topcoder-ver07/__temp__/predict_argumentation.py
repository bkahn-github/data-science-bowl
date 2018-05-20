from common import *
from utility.draw import *

from dataset.reader import *
from dataset.transform import *
from net.layer.mask_nms import *

##--------------------------------------------------------------
AUG_FACTOR = 16

def draw_proposal(image,proposal):
    image = image.copy()
    for p in proposal:
        x0,y0,x1,y1 = p[1:5].astype(np.int32)
        cv2.rectangle(image,(x0,y0),(x1,y1),(255,255,255),1)
    return image


def scale_to_factor(image, scale_x, scale_y, factor=16):
    height,width = image.shape[:2]
    h = math.ceil(scale_y*height/factor)*factor
    w = math.ceil(scale_x*width/factor)*factor
    image = cv2.resize(image,(w,h))
    return image


# def make_instance(net):
#     detection  = net.detections.data.cpu().numpy()
#     mask_logit = net.mask_logits.cpu().data.numpy()
#     mask_prob  = np_sigmoid(mask_logit)
#     mask       = net.masks[0]
#
#     height,width = mask.shape[:2]
#
#     instance =[]
#     num_detection = len(detection)
#     for n in range(num_detection):
#         m = np.zeros((height,width),np.float32)
#
#         _,x0,y0,x1,y1,score,label,k = detection[n]
#         x0 = int(round(x0))
#         y0 = int(round(y0))
#         x1 = int(round(x1))
#         y1 = int(round(y1))
#         label = int(label)
#         k = int(k)
#         h, w  = y1-y0+1, x1-x0+1
#
#         crop  = mask_prob[k, label]
#         crop  = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)
#         m[y0:y1+1,x0:x1+1] = crop
#         instance.append(m)
#
#     instance = np.array(instance,np.float32)
#     return instance


def undo_flip_transpose(height, width, image=None, proposal=None, mask=None, instance=None, type=0):
    undo=[
        (0, 1,0,  0,1,),
        (3, 0,1,  1,0,),
        (2, 1,0,  0,1,),
        (1, 0,1,  1,0,),
        (4, 1,0,  0,1,),
        (5, 1,0,  0,1,),
        (6, 0,1,  1,0,),
        (7, 0,1,  1,0,),
    ]
    t, m0, m1,m2, m3  = undo[type]
    H = m0*height + m1*width
    W = m2*height + m3*width
    return do_flip_transpose(H, W, image, proposal, mask, instance, t )


def do_flip_transpose(height, width, image=None, proposal=None, mask=None, instance=None, type=0):
    #choose one of the 8 cases

    if image    is not None: image=image.copy()
    if proposal is not None: proposal=proposal.copy()
    if mask     is not None: mask=mask.copy()
    if instance is not None: instance=instance.copy()

    if proposal is not None:
        x0 = proposal[:,1]
        y0 = proposal[:,2]
        x1 = proposal[:,3]
        y1 = proposal[:,4]

    if type==1: #rotate90
        if image is not None:
            image = image.transpose(1,0,2)
            image = cv2.flip(image,1)

        if mask is not None:
            #mask = np.rot90(mask,k=1)
            mask = mask.transpose(1,0)
            mask = np.fliplr(mask)


        if instance is not None:
            instance = instance.transpose(1,2,0)
            #instance = np.rot90(instance,k=1)
            instance = instance.transpose(1,0,2)
            instance = np.fliplr(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, y0, x1, y1  =  y0, x0,  y1, x1
            x0, x1 = height-1-x0, height-1-x1

    if type==2: #rotate180
        if image is not None:
            image = cv2.flip(image,-1)

        if mask is not None:
            mask = np.rot90(mask,k=2)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.rot90(instance,k=2)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, x1 = width -1-x0, width -1-x1
            y0, y1 = height-1-y0, height-1-y1

    if type==3: #rotate270
        if image is not None:
            image = image.transpose(1,0,2)
            image = cv2.flip(image,0)

        if mask is not None:
            #mask = np.rot90(mask,k=3)
            mask = mask.transpose(1,0)
            mask = np.flipud(mask)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            #instance = np.rot90(instance,k=3)
            instance = instance.transpose(1,0,2)
            instance = np.flipud(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, y0, x1, y1  =  y0, x0,  y1, x1
            y0, y1 = width-1-y0, width-1-y1

    if type==4: #flip left-right
        if image is not None:
            image = cv2.flip(image,1)

        if mask is not None:
            mask = np.fliplr(mask)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.fliplr(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, x1 = width -1-x0, width -1-x1

    if type==5: #flip up-down
        if image is not None:
            image = cv2.flip(image,0)

        if mask is not None:
            mask = np.flipud(mask)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.flipud(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            y0, y1 = height-1-y0, height-1-y1

    if type==6:
        if image is not None:
            image = cv2.flip(image,1)
            image = image.transpose(1,0,2)
            image = cv2.flip(image,1)

        if mask is not None:
            mask = cv2.flip(mask,1)
            mask = mask.transpose(1,0)
            mask = cv2.flip(mask,1)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.fliplr(instance)
            instance = np.rot90(instance,k=3)
            #instance = np.fliplr(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, x1 = width -1-x0, width -1-x1
            x0, y0, x1, y1  =  y0, x0,  y1, x1
            x0, x1 = height -1-x0, height -1-x1

    if type==7:
        if image is not None:
            image = cv2.flip(image,0)
            image = image.transpose(1,0,2)
            image = cv2.flip(image,1)

        if mask is not None:
            mask = cv2.flip(mask,0)
            mask = mask.transpose(1,0)
            mask = cv2.flip(mask,1)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.flipud(instance)
            instance = np.rot90(instance,k=3)
            #instance = np.fliplr(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            y0, y1 = height-1-y0, height-1-y1
            x0, y0, x1, y1  =  y0, x0,  y1, x1
            x0, x1 = height -1-x0, height -1-x1


    if proposal is not None:
        x0,x1 = np.minimum(x0,x1), np.maximum(x0,x1)
        y0,y1 = np.minimum(y0,y1), np.maximum(y0,y1)
        proposal[:,1] = x0
        proposal[:,2] = y0
        proposal[:,3] = x1
        proposal[:,4] = y1

    # if image    is not None: image=image.copy()
    # if proposal is not None: proposal=proposal.copy()
    # if mask     is not None: mask=mask.copy()
    # if instance is not None: instance=instance.copy()

    out=[]
    if image    is not None: out.append(image)
    if proposal is not None: out.append(proposal)
    if mask     is not None: out.append(mask)
    if instance is not None: out.append(instance)
    if len(out)==1: out=out[0]

    return out



## argument ##########################################

def do_test_augment_identity(image, proposal=None):
    height,width = image.shape[:2]
    h = math.ceil(height/AUG_FACTOR)*AUG_FACTOR
    w = math.ceil(width /AUG_FACTOR)*AUG_FACTOR
    dx = w-width
    dy = h-height

    image = cv2.copyMakeBorder(image, left=0, top=0, right=dx, bottom=dy,
                               borderType= cv2.BORDER_REFLECT101, value=[0,0,0] )

    if proposal is not None:
        h,w = image.shape[:2]
        proposal = proposal.copy()
        x1,y1 = proposal[:,3],proposal[:,4]
        x1[np.where(x1>width -1-dx)[0]]=w-1 #dx
        y1[np.where(y1>height-1-dy)[0]]=h-1 #dy
        proposal[:,3] = x1
        proposal[:,4] = y1
        return image, proposal

    else:
        return image


def undo_test_augment_identity(net, image):

    height,width = image.shape[:2]
    # h = math.ceil(height/AUG_FACTOR)*AUG_FACTOR
    # w = math.ceil(width /AUG_FACTOR)*AUG_FACTOR
    # dx = w-width
    # dy = h-height

    rcnn_proposal = net.rcnn_proposals.cpu().numpy()
    detection = net.detections.data.cpu().numpy()
    mask     = net.masks[0].copy()
    instance = net.mask_instances[0].copy()

    #rcnn_proposal = rcnn_proposal.copy()
    rcnn_proposal[:,1]=np.clip(rcnn_proposal[:,1],0,width -1)
    rcnn_proposal[:,2]=np.clip(rcnn_proposal[:,2],0,height-1)
    rcnn_proposal[:,3]=np.clip(rcnn_proposal[:,3],0,width -1)
    rcnn_proposal[:,4]=np.clip(rcnn_proposal[:,4],0,height-1)
    # ps = rcnn_proposal.copy()
    # rcnn_proposal=[]
    # for p in ps:
    #     i,x0,y0,x1,y1, score, label, aux = p
    #     x0 = max(min(x0,width -1),0)
    #     x1 = max(min(x1,width -1),0)
    #     y0 = max(min(y0,height-1),0)
    #     y1 = max(min(y1,height-1),0)
    #     w = x1-x0 + 1
    #     h = y1-y0 + 1
    #     if w>2 and h>2:
    #         rcnn_proposal.append([i,x0,y0,x1,y1, score, label, aux])
    # rcnn_proposal= np.array(rcnn_proposal, np.float32)

    #detection = detection.copy()
    detection[:,1]=np.clip(detection[:,1],0,width -1)
    detection[:,2]=np.clip(detection[:,2],0,height-1)
    detection[:,3]=np.clip(detection[:,3],0,width -1)
    detection[:,4]=np.clip(detection[:,4],0,height-1)

    # ps = detection.copy()
    # detection=[]
    # for t,p in enumerate(ps):
    #     i,x0,y0,x1,y1, score, label, aux = p
    #     x0 = max(min(x0,width -1),0)
    #     x1 = max(min(x1,width -1),0)
    #     y0 = max(min(y0,height-1),0)
    #     y1 = max(min(y1,height-1),0)
    #     w = x1-x0 + 1
    #     h = y1-y0 + 1
    #     if w>2 and h>2:
    #         detection.append([i,x0,y0,x1,y1, score, label, aux])
    #     else:
    #         mask[mask==t+1]=0
    # detection= np.array(detection, np.float32)

    mask     = mask[0:height,0:width]
    instance = instance[:, 0:height,0:width]

    return rcnn_proposal, detection, mask, instance


## argument ##########################################

def do_test_augment_flip_transpose(image, proposal=None, type=0):
    #image_show('image_before',image)
    height,width = image.shape[:2]
    image = do_flip_transpose(height,width, image=image, type=type)

    if proposal is not None:
        proposal = do_flip_transpose(height,width, proposal=proposal, type=type)

    return do_test_augment_identity(image, proposal)




def undo_test_augment_flip_transpose(net, image, type=0):
    height,width = image.shape[:2]
    undo=[
        (0, 1,0,  0,1,),
        (3, 0,1,  1,0,),
        (2, 1,0,  0,1,),
        (1, 0,1,  1,0,),
        (4, 1,0,  0,1,),
        (5, 1,0,  0,1,),
        (6, 0,1,  1,0,),
        (7, 0,1,  1,0,),
    ]
    t, m0, m1,m2, m3  = undo[type]
    H = m0*height + m1*width
    W = m2*height + m3*width

    dummy_image = np.zeros((H,W,3),np.uint8)
    rcnn_proposal, detection, mask, instance = undo_test_augment_identity(net, dummy_image)
    detection, mask, instance = do_flip_transpose(H,W, proposal=detection, mask=mask, instance=instance, type=t)
    rcnn_proposal             = do_flip_transpose(H,W, proposal=rcnn_proposal, type=t)

    return rcnn_proposal, detection, mask, instance



def do_test_augment_scale_flip_transpose(image, proposal=None, scale_x=1, scale_y=1, type=0):
    #image_show('image_before',image)
    height,width = image.shape[:2]
    image = do_flip_transpose(height,width, image=image, type=type)

    if proposal is not None:
        proposal = do_flip_transpose(height,width, proposal=proposal, type=type)

    return do_test_augment_scale(image, proposal, scale_x, scale_y)




def undo_test_augment_scale_flip_transpose(net, image, scale_x=1, scale_y=1, type=0):
    height,width = image.shape[:2]
    undo=[
        (0, 1,0,  0,1,),
        (3, 0,1,  1,0,),
        (2, 1,0,  0,1,),
        (1, 0,1,  1,0,),
        (4, 1,0,  0,1,),
        (5, 1,0,  0,1,),
        (6, 0,1,  1,0,),
        (7, 0,1,  1,0,),
    ]
    t, m0, m1,m2, m3  = undo[type]
    H = m0*height + m1*width
    W = m2*height + m3*width

    dummy_image = np.zeros((H,W,3),np.uint8)
    rcnn_proposal, detection, mask, instance = undo_test_augment_scale(net, dummy_image, scale_x, scale_y)
    detection, mask, instance = do_flip_transpose(H,W, proposal=detection, mask=mask, instance=instance, type=t)
    rcnn_proposal             = do_flip_transpose(H,W, proposal=rcnn_proposal, type=t)

    return rcnn_proposal, detection, mask, instance



## argument ##########################################

def do_test_augment_scale(image, proposal=None, scale_x=1, scale_y=1):
    height,width = image.shape[:2]

    image = scale_to_factor(image, scale_x, scale_y, factor=AUG_FACTOR)
    if proposal is not None:
        H,W = image.shape[:2]
        x0 = proposal[:,1]
        y0 = proposal[:,2]
        x1 = proposal[:,3]
        y1 = proposal[:,4]
        proposal[:,1] = np.round(x0 * (W-1)/(width -1))
        proposal[:,2] = np.round(y0 * (H-1)/(height-1))
        proposal[:,3] = np.round(x1 * (W-1)/(width -1))
        proposal[:,4] = np.round(y1 * (H-1)/(height-1))
        return image, proposal

    else :
        return image


def undo_test_augment_scale(net, image, scale_x=1, scale_y=1):

    def scale_mask(H,W, detection, mask_prob ):
        mask_threshold=0.5

        mask = np.zeros((H,W),np.int32)
        instance = []

        num_detection = len(detection)
        for n in range(num_detection):
            _,x0,y0,x1,y1,score,label,k = detection[n]
            x0 = int(round(x0))
            y0 = int(round(y0))
            x1 = int(round(x1))
            y1 = int(round(y1))
            label = int(label)
            k = int(k)
            h, w  = y1-y0+1, x1-x0+1

            crop  = mask_prob[k, label]
            crop  = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)

            m = np.zeros((H,W),np.float32)
            m[y0:y1+1,x0:x1+1] = crop
            instance.append(m.reshape(1,H,W))

            binary = instance_to_binary(m, threshold=0.5,min_area=5)
            mask[np.where(binary)] = n+1

        instance = np.vstack(instance)
        return mask, instance

    # ----
    rcnn_proposal = net.rcnn_proposals.cpu().numpy()
    detection     = net.detections.data.cpu().numpy()
    mask          =  net.masks[0]

    mask_logit = net.mask_logits.cpu().data.numpy()
    mask_prob  = np_sigmoid(mask_logit)


    height,width = image.shape[:2]
    H,W  = mask.shape[:2]
    scale_x = (width -1)/(W-1)
    scale_y = (height-1)/(H-1)

    x0 = rcnn_proposal[:,1]
    y0 = rcnn_proposal[:,2]
    x1 = rcnn_proposal[:,3]
    y1 = rcnn_proposal[:,4]
    rcnn_proposal[:,1] = np.round(x0 * scale_x)
    rcnn_proposal[:,2] = np.round(y0 * scale_y)
    rcnn_proposal[:,3] = np.round(x1 * scale_x)
    rcnn_proposal[:,4] = np.round(y1 * scale_y)

    x0 = detection[:,1]
    y0 = detection[:,2]
    x1 = detection[:,3]
    y1 = detection[:,4]
    detection[:,1] = np.round(x0 * scale_x)
    detection[:,2] = np.round(y0 * scale_y)
    detection[:,3] = np.round(x1 * scale_x)
    detection[:,4] = np.round(y1 * scale_y)

    mask, instance = scale_mask(height, width,  detection, mask_prob, )

    return rcnn_proposal, detection, mask, instance

# check #################################################################
def run_check_1():
    image_file='/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-05b/predict/tt_normal/overlays/8b59819fbc92eefe45b1db95c0cc3a467ddcfc755684c7f2ba2f6ccb9ad740ab/8b59819fbc92eefe45b1db95c0cc3a467ddcfc755684c7f2ba2f6ccb9ad740ab.mask.png'
    image = cv2.imread(image_file,cv2.IMREAD_COLOR)
    H,W = image.shape[:2]

    cv2.circle(image, (0,0),32,(255,255,255),-1)
    cv2.circle(image, (W-1,0),32,(0,255,255),-1)
    cv2.circle(image, (W-1,H-1),32,(255,255,0),-1)
    cv2.circle(image, (0,H-1),32,(255,0,255),-1)

    for t in range(8):
        print(t)
        image1 = do_flip_transpose(H,W,image, type=t)
        image2 = undo_flip_transpose(H,W,image1, type=t)

        image_show('image',image)
        image_show('image1',image1)
        image_show('image2',image2)
        cv2.waitKey(0)

def run_check_2():
    image_file='/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-05b/predict/tt_normal/overlays/8b59819fbc92eefe45b1db95c0cc3a467ddcfc755684c7f2ba2f6ccb9ad740ab/8b59819fbc92eefe45b1db95c0cc3a467ddcfc755684c7f2ba2f6ccb9ad740ab.mask.png'
    image = cv2.imread(image_file,cv2.IMREAD_COLOR)
    image = cv2.resize(image,dsize=(320,240))

    H,W = image.shape[:2]
    cv2.circle(image, (0,0),32,(255,255,255),-1)
    cv2.circle(image, (W-1,0),32,(0,255,255),-1)
    cv2.circle(image, (W-1,H-1),32,(255,255,0),-1)
    cv2.circle(image, (0,H-1),32,(255,0,255),-1)

    mask = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.int32)
    mask = relabel_mask(mask)
    box, label, instance = mask_to_annotation(mask, min_area=16, border=0, min_size=0, max_size=np.inf )

    proposal = np.zeros((len(box),8),np.float32)
    proposal[:,0  ]=0
    proposal[:,1:5]=box
    proposal[:,5  ]=1
    proposal[:,6  ]=label
    proposal[:,7  ]=0



    for t in range(8):
        print(t)
        image1, proposal1, mask1, instance1 = do_flip_transpose  (H,W,image,  proposal,  mask,  instance,  type=t)
        image2, proposal2, mask2, instance2 = undo_flip_transpose(H,W,image1, proposal1, mask1, instance1, type=t)

        image0 = draw_proposal(image, proposal )
        image1 = draw_proposal(image1,proposal1)
        image2 = draw_proposal(image2,proposal2)

        H1,W1 = image1.shape[:2]
        image_show('image',np.hstack([image0,image2]))
        image_show('image1',image1)
        #image_show('image2',image2)

        image_show('mask', 255*np.hstack([mask,mask2]))
        image_show('mask1',255*mask1)

        image_show('instance', 255*np.hstack([instance[:3].reshape(3*H,W),instance2[:3].reshape(3*H,W)]))
        image_show('instance1',255*instance1[:3].reshape(3*H1,W1))

        cv2.waitKey(0)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_1()
    #run_check_2()

    print('\nsucess!')



