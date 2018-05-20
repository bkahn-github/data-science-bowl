# reference:  https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/anchor_target_layer.py
from common import *
from utility.draw import*
from net.layer.box.process import*

if __name__ == '__main__':
    from rcnn_nms     import *
else:
    from .rcnn_nms    import *


def add_truth_box_to_proposal(cfg, proposal, b, truth_box, truth_label, score=-1):

    #proposal i,x0,y0,x1,y1,score, label
    if len(truth_box) !=0:
        truth = np.zeros((len(truth_box),8),np.float32)
        truth[:,0  ] = b
        truth[:,1:5] = truth_box
        truth[:,5  ] = score
        truth[:,6  ] = truth_label
        truth[:,7  ] = 0  #spare
    else:
        truth = np.zeros((0,8),np.float32)

    sampled_proposal = np.vstack([proposal,truth])
    return sampled_proposal


# mask target ********************************************************************
#<todo> mask crop should match align kernel (same wait to handle non-integer pixel location (e.g. 23.5, 32.1))
def crop_instance(instance, box, size, threshold=0.5):
    H,W = instance.shape
    x0,y0,x1,y1 = np.rint(box).astype(np.int32)
    x0 = max(0,x0)
    y0 = max(0,y0)
    x1 = min(W,x1)
    y1 = min(H,y1)

    #<todo> filter this
    if 1:
        if x0==x1:
            x0=x0-1
            x1=x1+1
            x0 = max(0,x0)
            x1 = min(W,x1)
        if y0==y1:
            y0=y0-1
            y1=y1+1
            y0 = max(0,y0)
            y1 = min(H,y1)

    #print(x0,y0,x1,y1)
    crop = instance[y0:y1+1,x0:x1+1]
    crop = cv2.resize(crop,(size,size))
    crop = (crop>threshold).astype(np.float32)
    return crop




# gpu version
## see https://github.com/ruotianluo/pytorch-faster-rcnn
def make_one_mask_rcnn_target(cfg, mode, input, proposal, truth_box, truth_label, truth_instance):
    rcnn_proposal = Variable(torch.FloatTensor((0,8))).cuda()
    rcnn_label    = Variable(torch.LongTensor ((0,1))).cuda()
    rcnn_target   = Variable(torch.FloatTensor((0,4))).cuda()
    mask_proposal = Variable(torch.FloatTensor((0,8))).cuda()
    mask_label    = Variable(torch.LongTensor ((0,1))).cuda()
    mask_instance = Variable(torch.FloatTensor(0,1,1)).cuda()

    if len(truth_box)==0 or len(proposal)==0:
        return rcnn_proposal, rcnn_label, rcnn_target, mask_proposal, mask_label, mask_instance


    #filter invalid proposal ---------------
    _,height,width = input.size()
    num_proposal = len(proposal)

    valid = []
    for i in range(num_proposal):
        box = proposal[i,1:5]
        if not(is_small_box(box, min_size=cfg.rcnn_train_min_size) ):  #is_small_box_at_boundary
            valid.append(i)

    if len(valid)==0:
        return rcnn_proposal, rcnn_label, rcnn_target, mask_proposal, mask_label, mask_instance

    proposal = proposal[valid]
    #----------------------------------------


    num_proposal = len(proposal)
    box = proposal[:,1:5]

    overlap = cython_box_overlap(box, truth_box)
    argmax_overlap = np.argmax(overlap,1)
    max_overlap = overlap[np.arange(num_proposal),argmax_overlap]

    fg_index = np.where( max_overlap >= cfg.rcnn_train_fg_thresh_low )[0]
    bg_index = np.where((max_overlap <  cfg.rcnn_train_bg_thresh_high) & \
                        (max_overlap >= cfg.rcnn_train_bg_thresh_low))[0]


    # sampling for class balance
    num_classes = cfg.num_classes
    num         = cfg.rcnn_train_batch_size
    num_fg      = int(np.round(cfg.rcnn_train_fg_fraction * cfg.rcnn_train_batch_size))

    # Small modification to the original version where we ensure a fixed number of regions are sampled
    # https://github.com/precedenceguo/mx-rcnn/commit/3853477d9155c1f340241c04de148166d146901d
    fg_length = len(fg_index)
    bg_length = len(bg_index)
    #print(fg_inds_length)

    if fg_length > 0 and bg_length > 0:
        num_fg = min(num_fg, fg_length)
        fg_index = fg_index[
            np.random.choice(fg_length, size=num_fg, replace=fg_length<num_fg)
        ]
        num_bg  = num - num_fg
        bg_index = bg_index[
            np.random.choice(bg_length, size=num_bg, replace=bg_length<num_bg)
        ]

    elif fg_length > 0:  #no bgs
        num_fg = num
        num_bg = 0
        fg_index = fg_index[
            np.random.choice(fg_length, size=num_fg, replace=fg_length<num_fg)
        ]

    elif bg_length > 0:  #no fgs
        num_fg = 0
        num_bg = num
        bg_index = bg_index[
            np.random.choice(bg_length, size=num_bg, replace=bg_length<num_bg)
        ]
        num_fg_proposal = 0
    else:
        # no bgs and no fgs?
        # raise NotImplementedError
        num_fg   = 0
        num_bg   = num
        bg_index = np.random.choice(num_proposal, size=num_bg, replace=num_proposal<num_bg)

    assert ((num_fg+num_bg)== num)


    # selecting both fg and bg
    index = np.concatenate([fg_index, bg_index], 0)
    sampled_proposal = proposal[index]

    #label
    sampled_assign = argmax_overlap[index]
    sampled_label  = truth_label[sampled_assign]
    sampled_label[num_fg:] = 0   # Clamp labels for the background to 0

    #target
    if num_fg>0:
        target_truth_box = truth_box[sampled_assign[:num_fg]]
        target_box       = sampled_proposal[:num_fg][:,1:5]
        sampled_target   = rcnn_encode(target_box, target_truth_box)


    #segment
    sampled_instance = []
    for i in range(num_fg):
        instance = truth_instance[sampled_assign[i]]
        box  = sampled_proposal[i,1:5]
        crop = crop_instance(instance, box, cfg.mask_size)
        sampled_instance.append(crop[np.newaxis,:,:])

        #<debug>
        if 0:
            print(sampled_label[i])
            x0,y0,x1,y1 = box.astype(np.int32)
            image = (instance*255 ).astype(np.uint8)
            #cv2.rectangle(image,(x0,y0),(x1,y1),128,1)
            image_show('image',image,2)
            image_show('crop',crop*255,2)
            cv2.waitKey(0)
    sampled_instance = np.vstack(sampled_instance)

    # save
    rcnn_proposal = Variable(torch.from_numpy(sampled_proposal)).cuda()
    rcnn_label    = Variable(torch.from_numpy(sampled_label)).long().cuda()
    rcnn_target   = Variable(torch.from_numpy(sampled_target)).cuda()

    mask_proposal = Variable(torch.from_numpy(sampled_proposal[:num_fg])).cuda()
    mask_label    = Variable(torch.from_numpy(sampled_label[:num_fg])).long().cuda()
    mask_instance = Variable(torch.from_numpy(sampled_instance)).cuda()

    return rcnn_proposal, rcnn_label, rcnn_target, mask_proposal, mask_label, mask_instance





def make_mask_rcnn_target(cfg, mode, inputs, proposals, truth_boxes, truth_labels, truth_instances):

    #<todo> take care of don't care ground truth. Here, we only ignore them  ----
    truth_boxes     = copy.deepcopy(truth_boxes)
    truth_labels    = copy.deepcopy(truth_labels)
    batch_size = len(inputs)
    for b in range(batch_size):
        index = np.where(truth_labels[b]>0)[0]
        truth_boxes [b] = truth_boxes [b][index]
        truth_labels[b] = truth_labels[b][index]
    #----------------------------------------------------------------------------


    proposals = proposals.cpu().data.numpy()
    rcnn_proposals = []
    rcnn_labels    = []
    rcnn_targets   = []
    mask_proposals = []
    mask_labels    = []
    mask_instances = []

    batch_size = len(truth_boxes)
    for b in range(batch_size):
        input          = inputs[b]
        truth_box      = truth_boxes[b]
        truth_label    = truth_labels[b]
        truth_instance = truth_instances[b]

        if len(truth_box) != 0:
            if len(proposals)==0:
                proposal = np.zeros((0,8),np.float32)
            else:
                proposal = proposals[proposals[:,0]==b]


            proposal = add_truth_box_to_proposal(cfg, proposal, b, truth_box, truth_label)


            rcnn_proposal, rcnn_label, rcnn_target, mask_proposal, mask_label, mask_instance = \
                make_one_mask_rcnn_target(cfg, mode, input, proposal, truth_box, truth_label, truth_instance)

            rcnn_proposals.append(rcnn_proposal)
            rcnn_labels.append(rcnn_label)
            rcnn_targets.append(rcnn_target)
            mask_proposals.append(mask_proposal)
            mask_labels.append(mask_label)
            mask_instances.append(mask_instance)


    rcnn_proposals = torch.cat(rcnn_proposals,0)
    rcnn_labels    = torch.cat(rcnn_labels,0)
    rcnn_targets   = torch.cat(rcnn_targets,0)
    mask_proposals = torch.cat(mask_proposals,0)
    mask_labels    = torch.cat(mask_labels,0)
    mask_instances = torch.cat(mask_instances,0)

    return rcnn_proposals, rcnn_labels, rcnn_targets, mask_proposals, mask_labels, mask_instances

################################################################################3



# cpu version
def make_one_mask_target(cfg, mode, input, proposal, truth_box, truth_label, truth_instance):

    sampled_proposal = Variable(torch.FloatTensor(0,8  )).cuda()
    sampled_label    = Variable(torch.LongTensor (0,1  )).cuda()
    sampled_instance = Variable(torch.FloatTensor(0,1,1)).cuda()

    if len(truth_box)==0 or len(proposal)==0:
        return sampled_proposal, sampled_label, sampled_instance


    #filter invalid proposal ---------------
    _,height,width = input.size()
    num_proposal = len(proposal)

    valid = []
    for i in range(num_proposal):
        box = proposal[i,1:5]
        if not(is_small_box(box, min_size=cfg.mask_train_min_size) ):  #is_small_box_at_boundary
            valid.append(i)

    if len(valid)==0:
        return sampled_proposal, sampled_label, sampled_instance

    proposal = proposal[valid]
    #----------------------------------------


    num_proposal = len(proposal)
    box = proposal[:,1:5]

    overlap = cython_box_overlap(box, truth_box)
    argmax_overlap = np.argmax(overlap,1)
    max_overlap = overlap[np.arange(num_proposal),argmax_overlap]
    fg_index = np.where( max_overlap >= cfg.mask_train_fg_thresh_low)[0]

    if len(fg_index)==0:
        return sampled_proposal, sampled_label, sampled_instance

    #<todo> sampling for class balance
    fg_length = len(fg_index)
    num_fg    = cfg.mask_train_batch_size
    fg_index  = fg_index[
        np.random.choice(fg_length, size=num_fg, replace=fg_length<num_fg)
    ]


    sampled_proposal = proposal[fg_index]
    sampled_assign   = argmax_overlap[fg_index]
    sampled_label    = truth_label[sampled_assign]
    sampled_instance = []
    for i in range(len(fg_index)):
        instance = truth_instance[sampled_assign[i]]
        box  = sampled_proposal[i,1:5]
        crop = crop_instance(instance, box, cfg.mask_size)
        sampled_instance.append(crop[np.newaxis,:,:])

        #<debug>
        if 0:
            print(sampled_label[i])
            x0,y0,x1,y1 = box.astype(np.int32)
            image = (instance*255 ).astype(np.uint8)
            cv2.rectangle(image,(x0,y0),(x1,y1),128,1)
            image_show('image',image,2)
            image_show('crop',crop*255,2)
            cv2.waitKey(0)
    sampled_instance = np.vstack(sampled_instance)

    # save
    sampled_proposal = Variable(torch.from_numpy(sampled_proposal)).cuda()
    sampled_label    = Variable(torch.from_numpy(sampled_label)).long().cuda()
    sampled_instance = Variable(torch.from_numpy(sampled_instance)).cuda()
    return sampled_proposal, sampled_label, sampled_assign, sampled_instance






def make_mask_target(cfg, mode, inputs, proposals, truth_boxes, truth_labels, truth_instances):

    #<todo> take care of don't care ground truth. Here, we only ignore them  ---
    truth_boxes     = copy.deepcopy(truth_boxes)
    truth_labels    = copy.deepcopy(truth_labels)
    truth_instances = copy.deepcopy(truth_instances)
    batch_size = len(inputs)
    for b in range(batch_size):
        index = np.where(truth_labels[b]>0)[0]
        truth_boxes [b] = truth_boxes [b][index]
        truth_labels[b] = truth_labels[b][index]
        truth_instances[b] = truth_instances[b][index]
    #----------------------------------------------------------------------------



    proposals = proposals.cpu().data.numpy()
    sampled_proposals  = []
    sampled_labels     = []
    sampled_assigns    = []
    sampled_instances  = []

    batch_size = len(truth_boxes)
    for b in range(batch_size):
        input          = inputs[b]
        truth_box      = truth_boxes[b]
        truth_label    = truth_labels[b]
        truth_instance = truth_instances[b]

        if len(truth_box) != 0:
            if len(proposals)==0:
                proposal = np.zeros((0,8),np.float32)
            else:
                proposal = proposals[proposals[:,0]==b]

            proposal = add_truth_box_to_proposal(cfg, proposal, b, truth_box, truth_label)
            sampled_proposal, sampled_label, sampled_assign, sampled_instance = \
                make_one_mask_target(cfg, mode, input, proposal, truth_box, truth_label, truth_instance)

            sampled_proposals.append(sampled_proposal)
            sampled_labels.append(sampled_label)
            sampled_assigns.append(sampled_assign)
            sampled_instances.append(sampled_instance)

    sampled_proposals = torch.cat(sampled_proposals,0)
    sampled_labels    = torch.cat(sampled_labels,0)
    sampled_instances = torch.cat(sampled_instances,0)

    return sampled_proposals, sampled_labels, sampled_assigns, sampled_instances



#-----------------------------------------------------------------------------  
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    check_layer()

 
 