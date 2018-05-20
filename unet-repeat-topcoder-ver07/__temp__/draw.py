from common import *
from net.metric import *
from dataset.reader import *

# if __name__ == '__main__':
#     from configuration import *
#     from layer.rpn_multi_nms     import *
#     from layer.rpn_multi_target  import *
#     from layer.rpn_multi_loss    import *
#
#
# else:
#     from .configuration import *
#     from .layer.rpn_multi_nms    import *
#     from .layer.rpn_multi_target import *
#     from .layer.rpn_multi_loss   import *





def draw_multi_rpn_prob(cfg, image, rpn_prob_flat):

    H,W = image.shape[:2]
    scales = cfg.rpn_scales
    num_scales = len(cfg.rpn_scales)
    num_bases  = [len(b) for b in cfg.rpn_base_apsect_ratios]

    rpn_prob = (rpn_prob_flat[:,1]*255).astype(np.uint8)
    rpn_prob = unflat_to_c3(rpn_prob, num_bases, scales, H, W)

    ## -pyramid -
    pyramid=[]
    for l in range(num_scales):
        pyramid.append(cv2.resize(image, None, fx=1/scales[l],fy=1/scales[l]))

    all = []
    for l in range(num_scales):
        a = np.vstack((
            pyramid[l],
            rpn_prob[l],
        ))

        all.append(
            cv2.resize(a, None, fx=scales[l],fy=scales[l],interpolation=cv2.INTER_NEAREST)
        )

    all = np.hstack(all).astype(np.uint8)
    draw_shadow_text(all,'rpn-prob', (5,15),0.5, (255,255,255), 1)

    return all



def draw_multi_rpn_delta(cfg, image, rpn_prob_flat, rpn_delta_flat, window, color=[255,255,255]):

    threshold = cfg.rpn_test_nms_pre_score_threshold

    image_box   = image.copy()
    image_point = image.copy()
    index = np.where(rpn_prob_flat>threshold)[0]
    for i in index:
        l = np.argmax(rpn_prob_flat[i])
        if l==0: continue

        w = window[i]
        t = rpn_delta_flat[i,l]
        b = rpn_decode(w.reshape(1,4), t.reshape(1,4))
        x0,y0,x1,y1 = b.reshape(-1).astype(np.int32)

        wx0,wy0,wx1,wy1 = window[i].astype(np.int32)

        cx = (wx0+wx1)//2
        cy = (wy0+wy1)//2

        #cv2.rectangle(image,(w[0], w[1]), (w[2], w[3]), (255,255,255), 1)
        cv2.rectangle(image_box,(x0,y0), (x1,y1), color, 1)
        image_point[cy,cx]= color


    draw_shadow_text(image_box,   'rpn-box', (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_point, 'point',   (5,15),0.5, (255,255,255), 1)
    all = np.hstack([image_box,image_point])
    return all


#pink
def draw_multi_rpn_proposal(cfg, image, proposal):

    image = image.copy()
    for p in proposal:
        x0,y0,x1,y1 = p[1:5].astype(np.int32)
        score = p[5]
        color = to_color(score, [255,0,255])
        cv2.rectangle(image, (x0,y0), (x1,y1), color, 1)

    return image



#yellow
def draw_truth_box(cfg, image, truth_box, truth_label):

    image = image.copy()
    if len(truth_box)>0:
        for b,l in zip(truth_box, truth_label):
            x0,y0,x1,y1 = b.astype(np.int32)
            if l <=0 : continue
            cv2.rectangle(image, (x0,y0), (x1,y1), [0,255,255], 1)

    return image


def draw_proposal_metric(image, proposal, truth_box, truth_label,
                          color0=[0,255,255], color1=[255,0,255], color2=[255,255,0],thickness=1):

    H,W = image.shape[:2]
    image_truth    = image.copy()  #yellow
    image_proposal = image.copy()  #pink
    image_hit      = image.copy()  #cyan
    image_miss     = image.copy()  #yellow
    image_fp       = image.copy()  #pink
    image_invalid  = image.copy()  #white
    precision = 0

    num_proposal = len(proposal)
    num_truth_box = len(truth_box)
    num_miss = 0
    num_fp   = 0
    percent_hit = 0

    if num_proposal>0 and num_truth_box>0:

        thresholds=[0.5,]

        box = proposal[:,1:5]
        precisions, recalls, results, truth_results = \
            compute_precision_for_box(box, truth_box, truth_label, thresholds)

        #for precision, recall, result, truth_result, threshold in zip(precisions, recalls, results, truth_results, thresholds):

        if 1:
            precision, recall, result, truth_result, threshold = \
                precisions[0], recalls[0], results[0], truth_results[0], thresholds[0]


            for i,b in enumerate(truth_box):
                x0,y0,x1,y1 = b.astype(np.int32)

                if truth_result[i]==HIT:
                    cv2.rectangle(image_truth,(x0,y0), (x1,y1), color0, thickness)
                    draw_screen_rect(image_hit,(x0,y0), (x1,y1), color2, 0.25)
                    percent_hit += 1

                if truth_result[i]==MISS:
                    cv2.rectangle(image_truth,(x0,y0), (x1,y1), color0, thickness)
                    cv2.rectangle(image_miss,(x0,y0), (x1,y1), color0, thickness)
                    num_miss += 1

                if truth_result[i]==INVALID:
                    draw_screen_rect(image_invalid,(x0,y0), (x1,y1), (255,255,255), 0.5)
            percent_hit = percent_hit/num_truth_box

            for i,b in enumerate(box):
                x0,y0,x1,y1 = b.astype(np.int32)
                cv2.rectangle(image_proposal,(x0,y0), (x1,y1), color1, thickness)

                if result[i]==TP:
                    cv2.rectangle(image_hit,(x0,y0), (x1,y1), color2, thickness)

                if result[i]==FP:
                    cv2.rectangle(image_fp,(x0,y0), (x1,y1), color1, thickness) #255,0,255
                    num_fp += 1

                if result[i]==INVALID:
                    cv2.rectangle(image_invalid,(x0,y0), (x1,y1), (255,255,255), thickness)


    draw_shadow_text(image_truth, 'truth(%d)'%num_truth_box,  (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_proposal,'proposal(%d)'%num_proposal, (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_hit, 'hit %0.2f'%percent_hit,  (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_miss,'miss(%d)'%num_miss, (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_fp,  'fp(%d)'%num_fp,   (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_invalid, 'n.a.', (5,15),0.5, (255,255,255), 1)

    all = np.hstack([image_truth,image_proposal,image_hit,image_miss,image_fp,image_invalid])
    draw_shadow_text(all,'%0.2f prec@0.5'%precision, (5,H-15),0.5, (255,255,255), 1)
    return all




def draw_mask_prob(mask_prob, mask_instance, mask_label):

    J=10
    I=5
    num,H,W = mask_instance.shape

    all = np.zeros((2*I*H, J*W, 3),np.uint8)
    for i in range(I):
        for j in range(J):

            n = i*J+j
            if n>num-1: continue

            if 1:
                t = mask_instance[n]>0.5
                m = mask_prob[n,1]>0.5
                hit  = t & m
                miss = t & (~m)
                fp   = (~t) & m
                overlap = np.zeros((H, W, 3),np.uint8)
                overlap[hit ]=[128,128,0]
                overlap[fp  ]=[255,0,255]
                overlap[miss]=[0,255,255]

                y0 = i*2*H
                y1 = y0 + H
                x0 = j*W
                x1 = x0 + W
                all[y0:y1,x0:x1] = overlap

            if 1:
                overlap = np.zeros((H, W, 3),np.uint8)
                overlap[:,:,1 ]=mask_instance[n]*255
                overlap[:,:,2 ]=mask_prob[n,1]*255

                y0 = (i*2+1)*H
                y1 = y0 + H
                x0 = j*W
                x1 = x0 + W
                all[y0:y1,x0:x1] = overlap



    return all


def draw_mask_metric(image, mask, truth_mask):
    mask       = relabel_mask(mask)
    truth_mask = relabel_mask(truth_mask)
    norm_image = do_gamma(image, 2.5)


    H,W = image.shape[:2]
    overlay_truth_mask = norm_image.copy()  #np.zeros((H,W,3),np.uint8)  #yellow
    overlay_mask       = norm_image.copy()  #np.zeros((H,W,3),np.uint8)  #pink
    overlay_hit        = np.zeros((H,W,3),np.uint8)
    overlay_miss       = norm_image.copy()  #np.zeros((H,W,3),np.uint8)  #yellow
    overlay_fp         = norm_image.copy()  #np.zeros((H,W,3),np.uint8)  #pink

    num_truth_mask = truth_mask.max()
    num_mask  = mask.max()
    num_hit  = 0
    num_miss = 0
    num_fp   = 0

    average_overlap   = 0
    average_precision = 0
    precision_50 = 0
    precision_70 = 0

    overlay_truth_mask = mask_to_contour_overlay(truth_mask, overlay_truth_mask, [0,255,255])
    overlay_mask       = mask_to_contour_overlay(mask, overlay_mask, [255,0,255])


    if num_truth_mask>0 and num_mask>0:

        #hit
        t = truth_mask!=0
        m = mask!=0
        hit  = t & m
        miss = t & (~m)
        fp   = (~t) & m
        overlay_hit[hit ]=[128,128,0]
        overlay_hit[fp  ]=[255,0,255]
        overlay_hit[miss]=[0,255,255]


        # metric -----
        predict     = mask
        truth       = truth_mask
        num_truth   = len(np.unique(truth  ))-1
        num_predict = len(np.unique(predict))-1
        assert(num_truth_mask==num_truth)
        assert(num_mask==num_predict,num_mask,num_predict)

        # intersection area
        intersection = np.histogram2d(truth.flatten(), predict.flatten(), bins=(num_truth+1, num_predict+1))[0]
        area_true = np.histogram(truth,   bins = num_truth  +1)[0]
        area_pred = np.histogram(predict, bins = num_predict+1)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred,  0)
        union = area_true + area_pred - intersection
        intersection = intersection[1:,1:]   # Exclude background from the analysis
        union = union[1:,1:]
        union[union == 0] = 1e-9
        iou = intersection / union   #iou = num_truth x num_predict


        precision = {}
        average_precision = 0
        thresholds = np.arange(0.5, 1.0, 0.05)
        for t in thresholds:
            tp, fp, fn = compute_precision(t, iou)
            prec = tp / (tp + fp + fn)
            precision[round(t,2) ]=prec
            average_precision += prec
        average_precision /= len(thresholds)
        precision_50 = precision[0.50]
        precision_70 = precision[0.70]


        #truth_mask
        overlap = np.max(iou,1)
        assign  = np.argmax(iou,1)
        average_overlap = overlap.mean()

        for t in range(num_truth):
            m = truth_mask==t+1
            contour = mask_to_inner_contour(m)
            s = overlap[t]
            if s>0.5:
                #color = to_color(max(0.0,(overlap[t]-0.5)/0.5), [255,255,255])
                #overlay_metric[m]=color
                pass
            else:
                #color = [0,0,255] #to_color(max(0.0,(0.5-overlap[t])/0.5), [0,255,0])
                #overlay_metric[m]=color
                overlay_miss[contour]=[0,255,255]
                num_miss = num_miss+1

        #mask
        overlap = np.max(iou,0)
        assign  = np.argmax(iou,0)
        for t in range(num_predict):
            m = mask==t+1
            contour = mask_to_inner_contour(m)
            s = overlap[t]
            if s>0.5:
                pass
            else:
                overlay_fp[contour]=[255,0,255]
                num_fp = num_fp+1

                # print(s)
                # image_show('overlay_metric',overlay_metric)
                # cv2.waitKey(0)

    #mask_score = cv2.cvtColor((np.clip(mask_score,0,1)*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    #mask_score = cv2.cvtColor((mask_score/mask_score.max()*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)

    draw_shadow_text(overlay_truth_mask,  'truth(%d)'%num_truth_mask,  (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(overlay_mask,        'mask(%d)'%num_mask,   (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(overlay_hit,         '%0.2f iou '%average_overlap, (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(overlay_miss,        'miss(%d)'%num_miss,  (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(overlay_fp,          'fp(%d)'%num_fp,    (5,15),0.5, (255,255,255), 1)

    all = np.hstack((overlay_truth_mask, overlay_mask, overlay_hit, overlay_miss, overlay_fp, image, ))
    draw_shadow_text(all,'%0.2f prec@0.5'%(precision_50), (5,H-45),0.5, (255,255,255), 1)
    draw_shadow_text(all,'%0.2f prec@0.7'%(precision_70), (5,H-30),0.5, (255,255,255), 1)
    draw_shadow_text(all,'%0.2f prec'%average_precision,  (5,H-15),0.5, (255,255,0), 1)
    #image_show('all mask : image, truth, predict, error, metric',all,1)

    return all


def draw_predict_mask(threshold, image, mask, detection):

    H,W = image.shape[:2]
    norm_image = do_gamma(image,2.5)

    box_overlay     = norm_image.copy()
    contour_overlay = norm_image.copy()
    color_overlay   = np.zeros((H,W,3),np.uint8)

    if len(detection)>0:
        colors = matplotlib.cm.get_cmap('hot')
        multi_mask = mask

        for i,d in enumerate(detection):
            mask = (multi_mask==i+1)
            contour = mask_to_inner_contour(mask)
            score   = d[5]
            s       = max(0,(score-threshold)/(1-threshold))

            color = colors(s)   #(0,1,0)  #
            color = int(color[2]*255),int(color[1]*255),int(color[0]*255)
            contour_overlay[contour] = color
            color_overlay[mask]      = color
            color_overlay[contour]   = [255,255,255]

            x0,y0,x1,y1 = d[1:5].astype(np.int32)
            cv2.rectangle(box_overlay, (x0,y0), (x1,y1), color, 1)

    all = np.hstack([image, color_overlay, contour_overlay, box_overlay])
    draw_shadow_text(all, 'threshold=%0.3f'%threshold,  (5,15),0.5, (255,255,255), 1)

    return all




def draw_predict_proposal(threshold, image, proposal):

    H,W = image.shape[:2]
    box_overlay = do_gamma(image,2.5)

    if len(proposal)>0:
        colors = matplotlib.cm.get_cmap('hot')

        for i,d in enumerate(proposal):
            score   = d[5]
            s       = max(0,(score-threshold)/(1-threshold))

            color = colors(s)
            color = int(color[2]*255),int(color[1]*255),int(color[0]*255)  #(0,0,255)  #

            x0,y0,x1,y1 = d[1:5].astype(np.int32)
            cv2.rectangle(box_overlay, (x0,y0), (x1,y1), color, 1)

    all = np.hstack([image, box_overlay])
    draw_shadow_text(all, 'threshold=%0.3f'%threshold,  (5,15),0.5, (255,255,255), 1)

    return all