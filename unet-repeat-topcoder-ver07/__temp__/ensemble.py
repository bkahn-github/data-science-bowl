
from common import *
from dataset.reader import *
from net.layer.box.process import *


#post filtering -------------------

#https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
def remove_holes(bin):
    bin_mask = np.copy(bin)
    h,w = bin.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(bin_mask,mask,(0,0),1)
    binmask_inv = cv2.bitwise_not(bin_mask)
    return bin | binmask_inv

#https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
#https://stackoverflow.com/questions/14854592/retrieve-elongation-feature-in-python-opencv-what-kind-of-moment-it-supposed-to
def is_long_fibre(instance):

    binary = instance>0.5
    contour = cv2.findContours(binary.astype(np.uint8)*255,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
    m = cv2.moments(contour[0])

    #elongation
    x = m['mu20'] + m['mu02']
    y = 4 * m['mu11']**2 + (m['mu20'] - m['mu02'])**2
    elongation = (x + y**0.5) / (x - y**0.5 +1e-12)

    area = binary.sum()
    is_long_fibre =  elongation>15 and area>4000
    #print(elongation,area)

    return is_long_fibre



#----------------------------------------------------------

def instance_to_box(instance, threshold=0.5):
    m = instance>threshold
    y,x = np.where(m)
    y0 = y.min()
    y1 = y.max()
    x0 = x.min()
    x1 = x.max()
    return x0,y0,x1,y1


def is_border_instance(instance):
    height,width = instance.shape[:2]

    x0,y0,x1,y1 = instance_to_box(instance)
    w = x1 - x0
    h = y1 - y0
    x = (x1 + x0)/2
    y = (y1 + y0)/2


    is_border = 0
    if \
       (x0<w/4          and h/w>3.5) or \
       (x1>width-1-w/4  and h/w>3.5) or \
       (y0<h/4          and w/h>3.5) or \
       (y1>height-1-h/4 and w/h>3.5) or \
       0: is_border=1

    return is_border

# def is_outlier_size(instances):
#     height,width  = instance[0].shape[:2]
#     num_instances = len(instances)
#     if num_instances<5: return []
#
#     sizes=[]
#     for i in range(num_instances):
#         m = instances[i]
#         x0,y0,x1,y1 = instance_to_box(m)
#         w = x1 - x0
#         h = y1 - y0
#         area = (m>0.5).sum()
#         sizes.append((w,h,area))
#     sizes = np.array(sizes)
#     median_w, median_h, median_area = np.median(sizes)
#
#     #<todo>  ---------------------
#
#     return []


#ensemble =======================================================

class Cluster(object):
    def __init__(self):
        super(Cluster, self).__init__()
        self.members=[]
        self.center =[]

    def add_item(self, proposal, instance, ):

        # instance = self.process_funct(instance)
        if self.center ==[]:
            self.members = [{
                'proposal': proposal, 'instance': instance
            },]
            self.center  = {
                'union_proposal': proposal, 'union_instance':instance,
                'inter_proposal': proposal, 'inter_instance':instance,
                'average_proposal': proposal, 'average_instance':instance,
            }
        else:
            N = len(self.members)
            self.members.append({
                'proposal': proposal, 'instance': instance
            })
            center_union_proposal = self.center['union_proposal'].copy()
            center_union_instance = self.center['union_instance'].copy()
            center_inter_proposal = self.center['inter_proposal'].copy()
            center_inter_instance = self.center['inter_instance'].copy()
            center_average_proposal = self.center['average_proposal'].copy()
            center_average_instance = self.center['average_instance'].copy()


            self.center['union_proposal'][1] = max(center_union_proposal[1],proposal[1])
            self.center['union_proposal'][2] = max(center_union_proposal[2],proposal[2])
            self.center['union_proposal'][3] = min(center_union_proposal[3],proposal[3])
            self.center['inter_proposal'][4] = min(center_union_proposal[4],proposal[4])
            self.center['union_proposal'][5] = min(center_union_proposal[5],proposal[5])
            self.center['union_instance'] = np.maximum(center_union_instance , instance )


            self.center['inter_proposal'][1] = max(center_inter_proposal[1],proposal[1])
            self.center['inter_proposal'][2] = max(center_inter_proposal[2],proposal[2])
            self.center['inter_proposal'][3] = min(center_inter_proposal[3],proposal[3])
            self.center['inter_proposal'][4] = min(center_inter_proposal[4],proposal[4])
            self.center['inter_proposal'][5] = min(center_inter_proposal[5],proposal[5])
            self.center['inter_instance'] = np.minimum(center_inter_instance , instance )

            self.center['inter_proposal'][1:6] = (N*center_average_proposal[1:6] + proposal[1:6]  )/(N+1)
            self.center['average_instance']    = (N*center_average_instance + instance  )/(N+1)


    def compute_iou(self, proposal, instance, type='union'):

        if type=='union':
            center_proposal = self.center['union_proposal']
            center_instance = self.center['union_instance']
        elif type=='inter':
            center_proposal = self.center['inter_proposal']
            center_instance = self.center['inter_instance']
        elif type=='average':
            center_proposal = self.center['average_proposal']
            center_instance = self.center['average_instance']
        else:
            raise NotImplementedError

        x0 = int(max(proposal[1],center_proposal[1]))
        y0 = int(max(proposal[2],center_proposal[2]))
        x1 = int(min(proposal[3],center_proposal[3]))
        y1 = int(min(proposal[4],center_proposal[4]))

        w = max(0,x1-x0)
        h = max(0,y1-y0)
        box_intersection = w*h
        if box_intersection<0.01: return 0

        x0 = int(min(proposal[1],center_proposal[1]))
        y0 = int(min(proposal[2],center_proposal[2]))
        x1 = int(max(proposal[3],center_proposal[3]))
        y1 = int(max(proposal[4],center_proposal[4]))

        i0 = center_instance[y0:y1,x0:x1]>0.5  #center_inter[y0:y1,x0:x1]
        i1 = instance[y0:y1,x0:x1]>0.5

        intersection = np.minimum(i0, i1).sum()
        area    = np.maximum(i0, i1).sum()
        overlap = intersection/(area + 1e-12)

        if 0: #debug
            m = np.zeros((*instance.shape,3),np.uint8)
            m[:,:,0]=instance*255
            m[:,:,1]=center_instance*255

            cv2.rectangle(m, (x0,y0),(x1,y1),(255,255,255),1)
            image_show('m',m)
            print('%0.5f'%overlap)
            cv2.waitKey(0)

        return overlap

# def sort_clsuters(clusters):
#
#     value=[]
#     for c in clusters:
#         x0,y0,x1,y1 = (c.center['inter_proposal'] + c.center['union_proposal'])[1:5]
#         value.append((x0+x1)+(y0+y1)*100000)
#     value=np.array(value)
#     index = list(np.argsort(value))
#
#     return index




def run_ensemble():

    out_dir = \
        RESULTS_DIR + '/stage_2/ensemble'
        #'/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-05d/ensemble/002'

    ensemble_dirs = [
        RESULTS_DIR + '/stage_2/predict/%s'%e
        #'/root/share/project/kaggle/science2018/results/mask-se-resnext50-rcnn_2crop-mega-05d/predict/%s'%e

        for e in [
            'xx_gray_normal',
            'xx_gray_flip_transpose_1',
            'xx_gray_flip_transpose_2',
            'xx_gray_flip_transpose_3',
            'xx_gray_flip_transpose_4',
            'xx_gray_flip_transpose_5',
            'xx_gray_flip_transpose_6',
            'xx_gray_flip_transpose_7',
            # #
            'xx_gray_scale_0.8',
            'xx_gray_scale_1.2',
            'xx_gray_scale_0.5',
            'xx_gray_scale_1.8',


            # 57
            # 'xx57_scale_0.8_flip_transpose_1',
            # 'xx57_scale_0.8_flip_transpose_2',
            # 'xx57_scale_0.8_flip_transpose_3',
            # 'xx57_scale_0.8_flip_transpose_4',
            # 'xx57_scale_0.8_flip_transpose_5',
            # 'xx57_scale_0.8_flip_transpose_6',
            # 'xx57_scale_0.8_flip_transpose_7',
            # #  57
            # 'xx57_scale_1.2_flip_transpose_1',
            # 'xx57_scale_1.2_flip_transpose_2',
            # 'xx57_scale_1.2_flip_transpose_3',
            # 'xx57_scale_1.2_flip_transpose_4',
            # 'xx57_scale_1.2_flip_transpose_5',
            # 'xx57_scale_1.2_flip_transpose_6',
            # 'xx57_scale_1.2_flip_transpose_7',
            # #  57
            # 'xx57_stretch_0.5_flip_transpose_1',
            # 'xx57_stretch_0.5_flip_transpose_2',
            # 'xx57_stretch_0.5_flip_transpose_3',
            # 'xx57_stretch_0.5_flip_transpose_4',
            # 'xx57_stretch_0.5_flip_transpose_5',
            # 'xx57_stretch_0.5_flip_transpose_6',
            # 'xx57_stretch_0.5_flip_transpose_7',
    ]]


    # names = glob.glob(ensemble_dirs[0] + '/overlays/*/')
    # names = [n.split('/')[-2]for n in names]
    # sorted(names)

    split = 'test2_ids_gray_2800'  #'BBBC006'   #'valid1_ids_gray2_43'
    ids   = read_list_from_file(DATA_DIR + '/split/' + split, comment='#') [:1445] #[:800]  #[:10]  #  [10:] #try 10 images for debug


    #########################
    # clustering parameters
    INSTANCE_MIN_AREA=5
    INSTANCE_MAX_AREA_RATIO=0.20
    INSTANCE_BASE_PROB=0.01
    INSTANCE_HIGH_PROB=0.80
    INSTANCE_CLUSTER_REJECT_SCORE=0.10
    INSTANCE_CLUSTER_IOU_ACCEPT=0.20
    INSTANCE_CLUSTER_IOU_REJECT=0.10
    INSTANCE_CLUSTER_MIN_NUM_MEMBERS=0.10



    #setup ---------------------------------------
    os.makedirs(out_dir +'/overlays', exist_ok=True)
    os.makedirs(out_dir +'/masks', exist_ok=True)


    num_ensemble = len(ensemble_dirs)
    for i in range(len(ids)):
        folder, name = ids[i].split('/')[-2:]

        # test post processing on problems images
        #name = '1962d0c5faf3e85cda80e0578e0cb7aca50826d781620e5c1c4cc586bc69f81a'
        #name = '38f5cfb55fc8b048e82a5c895b25fefae7a70c71ab9990c535d1030637bf6a1f'
        #name = '0e132f71c8b4875c3c2dd7a22997468a3e842b46aa9bd47cf7b0e8b7d63f0925'
        #name = '4f949bd8d914bbfa06f40d6a0e2b5b75c38bf53dbcbafc48c97f105bee4f8fac'
        #name = 'eea70a7948d25a9a791dbcb39228af4ea4049fe5ebdee9c04884be8cca3da835'


        print('%05d %s'%(i,name))
        image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png'%(folder,name), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        #image_show('image',image,1)

        #load all first, filter invalid case ---
        instances=[]
        proposals=[]
        scores=[]
        for dir in ensemble_dirs:
            instance = np.load(dir +'/instances/%s.npy'%(name)).astype(np.float32)/255
            proposal = np.load(dir +'/detections/%s.npy'%(name))
            assert(len(proposal)==len(instance))
            #print(len(instance),len(proposal))

            num = len(instance)
            for t in range(num):
                p, m = proposal[t],instance[t]
                area = (m>0.5).sum()
                if area <INSTANCE_MIN_AREA: continue
                if area >INSTANCE_MAX_AREA_RATIO*height*width: continue


                #compute score
                area = (m>INSTANCE_BASE_PROB).sum()
                high = (m>INSTANCE_HIGH_PROB).sum()
                score = high/area

                p[-1]=score
                instances.append(m)
                proposals.append(p)
                scores.append(score)

                # if (m>0.01).sum()>10000:
                #     print(score)
                #     print((m>0.01).sum()/(height*width))
                #     image_show_norm('m', m, max=1,resize=1)
                #     cv2.waitKey(0)
                #     zz=0

        #flatten
        scores    = np.array(scores)
        indices   = np.argsort(-scores)
        instances = np.array(instances)[indices]
        proposals = np.array(proposals)[indices]
        scores    = np.array(scores)[indices]
        num_instances = len(instances)

        # this is to normalise score
        all_instances = np.zeros((height, width), np.float32)
        for t in range(num_instances):
            all_instances += instances[t]


        ##fix for zero nuclei
        if all_instances.sum()==0: continue

        strength = np.percentile(all_instances[np.where(all_instances>0.1)], 90) #all_instances.max()
            #np.percentile(all_instances[np.where(all_instances>0.1)], 0.9)


        zz=0
        #show
        if 0: #is_debug:
            for t in range(num_instances):
                print(t, scores[t])
                image_show_norm('num_instances', instances[t], resize=1)
                cv2.waitKey(0)


        #------------------------------------------------------------------
        #cluster high confidence ones first

        clusters = []
        c = Cluster()
        c.add_item(proposals[0],instances[0])
        clusters.append(c)

        reject=[]
        for t in range(1, num_instances):
            if scores[t]<INSTANCE_CLUSTER_REJECT_SCORE :  ###adjust this!!!! INSTANCE_CLUSTER_REJECT_SCORE
                reject.append(t)
                continue

            p, m = proposals[t], instances[t]
            max_iou=0
            if len(clusters)!=0:
                ious = np.array( [c.compute_iou(p, m, type='average') for c in clusters], np.float32 )
                k = np.argmax(ious)
                max_iou = ious[k]

                if max_iou>INSTANCE_CLUSTER_IOU_ACCEPT:
                    c = clusters[k]
                    c.add_item(p, m)

                elif max_iou<INSTANCE_CLUSTER_IOU_REJECT:
                    c = Cluster()
                    c.add_item(p, m)
                    clusters.append(c)

                else:
                    reject.append(t)

        num_clusters= len(clusters)
        #print(num_clusters)

        #make submit mask here!! ----------------------------------------------
        foreground = np.zeros((height, width), np.int32)
        for k in reversed(range(num_clusters)):
            center = clusters[k].center['average_instance']
            binary = center>0.5
            foreground += binary


        mask = np.zeros((height, width), np.int32)  #do backwards
        id=1
        for k in reversed(range(num_clusters)):
            print (k)
            center = clusters[k].center['average_instance']
            binary = center>0.5






            # print(is_border_instance(binary))
            # print('\t',len(clusters[k].members))
            # print(binary.sum())
            # image_show_norm('binary', binary, resize=1)
            # image_show_norm('mask', mask, resize=1)
            # cv2.waitKey(0)

            #<todo>
            #if len(clusters[k].members)<INSTANCE_CLUSTER_MIN_NUM_MEMBERS*num_ensemble: continue
            num_members = len(clusters[k].members)
            s = center[binary].sum()/binary.sum()*num_members
            if s<4: continue
            #if s<1: continue  #chnage this according to num of ensembles

            if is_border_instance(binary)==1: continue
            if is_long_fibre(binary)==1: continue
            #binary = remove_holes(binary)

            # image_show_norm('binary', binary, resize=1)
            # image_show_norm('mask', mask, resize=1)
            # cv2.waitKey(0)

            # <no time todo>
            # refine reject instances (split small)
            # filter small
            # filter dark?

            # filter unsual size
            # smooth jagged shape
            # extend to border

            #choose max score
            # num_members = len(clusters[k].members)
            # member_scores = np.array([m['proposal'][-1] for m in clusters[k].members])
            # j = np.argmax(member_scores)
            # print (clusters[k].members[j]['proposal'][-1])
            # binary = clusters[k].members[j]['instance']>0.5

            mask[binary]=id
            id+=1

        print('id :',id)
        np.save(out_dir +'/masks/%s.npy'%(name),mask)



        #show
        if 1: #is_debug:

            reject_instances = np.zeros((height, width), np.float32)
            for t in reject:
                reject_instances += instances[t]




            display1 = do_gamma(image,2.5)  #image.copy()
            display2 = np.zeros((height, width), np.float32)
            for k in range(num_clusters):
                #print (k)

                center = clusters[k].center['average_instance']
                binary = center>0.5

                display2 += binary
                display1[mask_to_inner_contour(binary)] = (255,255,0)

                # image_show_norm('center', center, resize=1)
                # image_show_norm('display1', display1, resize=1)
                # image_show_norm('display2', display2, resize=1)
                # cv2.waitKey(0)


            color_overlay = mask_to_color_overlay(mask)
            color1_overlay  = mask_to_contour_overlay(mask, color_overlay)
            contour_overlay = mask_to_contour_overlay(mask,do_gamma(image,2.5), (0,255,0))

            display3 = to_color_image(reject_instances, max=None)
            display4 = to_color_image(all_instances, max=None)
            display2 = to_color_image(display2, max=None)
            all= np.hstack([image, color1_overlay, contour_overlay, display1, display2, display3, display4])
            image_show_norm('all', all, resize=1)

            cv2.imwrite(out_dir +'/overlays/%s.png'%(name),all)


        cv2.waitKey(1)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_ensemble()
    print('\nsucess!')