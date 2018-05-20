from common import *
from net.layer.box.process import*
from utility.draw import *




def make_empty_masks(cfg, mode, inputs):#<todo>
    masks = []
    batch_size,C,H,W = inputs.size()
    for b in range(batch_size):
        mask = np.zeros((H, W), np.float32)
        masks.append(mask)
    return masks

def make_empty_mask_instances(cfg, mode, inputs):#<todo>
    mask_instances = []
    batch_size,C,H,W = inputs.size()
    for b in range(batch_size):
        mask_instance = np.zeros((0,H, W), np.float32)
        mask_instances.append(mask_instance)
    return mask_instances


#https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def mask_to_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return x0, y0, x1, y1



def instance_to_binary(instance, threshold, min_area):

    #remove fragments---
    binary = instance>threshold
    label  = skimage.morphology.label(binary)
    num_labels = label.max()
    if num_labels>0:
        areas    = [(label==c+1).sum() for c in range(num_labels)]
        max_area = max(areas)

        for c in range(num_labels):
            if areas[c] != max_area:
                binary[label==c+1]=0
            else:
                if max_area<min_area:
                    binary[label==c+1]=0

    #<todo> fill holes? ---

    return binary


#--------------------------------
#  one to one correspondemce: (proposals, mask_logits)
#                                      |
#                                      V
#                             (masks, mask_instances, mask_proposals)
#
#



def mask_nms( cfg, mode, inputs, proposals, mask_logits):
    assert(len(proposals)==len(mask_logits))

    overlap_threshold   = cfg.mask_test_nms_overlap_threshold
    pre_score_threshold = cfg.mask_test_nms_pre_score_threshold
    mask_threshold      = cfg.mask_test_mask_threshold
    mask_min_area       = cfg.mask_test_mask_min_area

    proposals   = proposals.cpu().data.numpy()
    mask_logits = mask_logits.cpu().data.numpy()
    mask_probs  = np_sigmoid(mask_logits)

    masks = []
    mask_proposals = []
    mask_instances = []
    batch_size,C,H,W = inputs.size()

    for b in range(batch_size):
        mask = np.zeros((H,W),np.int32)
        mask_proposal = []
        mask_instance = []
        num_keeps = 0

        index = np.where((proposals[:,0]==b) & (proposals[:,5]>pre_score_threshold))[0]
        if len(index) != 0:
            instance = []
            box      = []
            for i in index:
                m = np.zeros((H,W),np.float32)

                x0,y0,x1,y1 = proposals[i,1:5].astype(np.int32)
                h, w  = y1-y0+1, x1-x0+1
                label = int(proposals[i,6])
                crop  = mask_probs[i, label]
                crop  = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)

                #-----------------------------------------------
                if 0: #<debug>
                    image = (inputs.data.cpu().numpy().transpose((0,2,3,1))*255).astype(np.uint8)[b].copy()
                    cv2.rectangle(image,(x0,y0),(x1,y1),(0,0,255),1)
                    image_show('image',image,1)
                    image_show('crop',crop.astype(np.float32)*255,1)
                    cv2.waitKey(0)
                #-----------------------------------------------
                m[y0:y1+1,x0:x1+1] = crop

                instance.append(m)
                box.append([x0,y0,x1,y1])

            # compute overlap ## ================================
            L = len(index)
            binary = [instance_to_binary(m, mask_threshold,mask_min_area) for m in instance]

            box              = np.array(box, np.float32)
            box_overlap      = cython_box_overlap(box, box)
            instance_overlap = np.zeros((L,L),np.float32)
            for i in range(L):
                instance_overlap[i,i] = 1
                for j in range(i+1,L):
                    if box_overlap[i,j]<0.01: continue

                    x0 = int(min(box[i,0],box[j,0]))
                    y0 = int(min(box[i,1],box[j,1]))
                    x1 = int(max(box[i,2],box[j,2]))
                    y1 = int(max(box[i,3],box[j,3]))

                    mi = binary[i][y0:y1,x0:x1]
                    mj = binary[j][y0:y1,x0:x1]

                    intersection = (mi & mj).sum()
                    area = (mi | mj).sum()
                    instance_overlap[i,j] = intersection/(area + 1e-12)
                    instance_overlap[j,i] = instance_overlap[i,j]

            #non-max suppress
            score = proposals[index,5]
            sort  = list(np.argsort(-score))

            ##  https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
            keep = []
            while len(sort) > 0:
                i = sort[0]
                keep.append(i)
                delete_index = [i] + list(np.where(instance_overlap[i] > overlap_threshold)[0])
                sort =  [e for e in sort if e not in delete_index]

            num_keeps=len(keep)
            for i in range(num_keeps):
                k = keep[i]
                mask[np.where(binary[k])] = i+1
                mask_instance.append(instance[k].reshape(1,H,W))

                t = index[k]
                b,x0,y0,x1,y1,score,label,_ = proposals[t]
                mask_proposal.append(np.array([b,x0,y0,x1,y1,score,label,t],np.float32))


        if num_keeps==0:
            mask_proposal = np.zeros((0,8  ),np.float32)
            mask_instance = np.zeros((0,H,W),np.float32)
        else:
            mask_proposal = np.vstack(mask_proposal)
            mask_instance = np.vstack(mask_instance)

        mask_proposals.append(mask_proposal)
        mask_instances.append(mask_instance)
        masks.append(mask)

        #print(multi_mask.max(),len(mask_proposal), keep)
        #assert(multi_mask.max()==len(mask_proposal))

    mask_proposals = Variable(torch.from_numpy(np.vstack(mask_proposals))).cuda()
    return masks, mask_instances, mask_proposals

##-----------------------------------------------------------------------------  
#if __name__ == '__main__':
#    print( '%s: calling main function ... ' % os.path.basename(__file__))
#
#
#
# 
 
