import os, sys
sys.path.append(os.path.dirname(__file__))

from train import *

#--------------------------------------------------------------
def label_to_mask(label):
    binary   = label>0
    marker   = skimage.morphology.label(label==1)
    distance = -label
    water = skimage.morphology.watershed(-distance, marker, connectivity=1, mask=binary)

    return water




#--------------------------------------------------------------
AUG_FACTOR = 16

def do_test_augment_identity(image):
    height,width = image.shape[:2]
    h = math.ceil(height/AUG_FACTOR)*AUG_FACTOR
    w = math.ceil(width /AUG_FACTOR)*AUG_FACTOR
    dx = w-width
    dy = h-height

    image = cv2.copyMakeBorder(image, left=0, top=0, right=dx, bottom=dy,
                               borderType= cv2.BORDER_REFLECT101, value=[0,0,0] )


    return image


def undo_test_augment_identity(net, image):
    height,width = image.shape[:2]

    prob  = np_softmax(net.logits.data.cpu().numpy())[0]
    label = np.argmax(prob,0).astype(np.float32)
    mask  = label_to_mask(label)

    mask = mask[:height, :width]
    return mask



def predict_one_image(image,net):



    pass








def run_predict():

    out_dir = RESULTS_DIR + '/unet-se-resnext50-7'
    initial_checkpoint = \
    RESULTS_DIR + '/unet-se-resnext50-7/checkpoint/00136400_model.pth'
        #
    # augment -----------------------------------------------------------------------------------------------------
    augments=[
        #tag_name, do_test_augment, undo_test_augment, params
        ('identity', do_test_augment_identity, None, {}),
    ]

    #-----------------------------------------------------------------------------------

    split = 'test1_ids_gray2_53'  #'valid1_ids_gray2_38'  #'BBBC006'
    ids   = read_list_from_file(DATA_DIR + '/split/' + split, comment='#') #[2507:] #[:800]  #[:10]  #  [10:] #try 10 images for debug



    #start experiments here! ###########################################################
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.evaluate.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## net ------------------------------
    cfg = Configuration()
    net = Net(cfg).cuda()
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n\n'%(type(net)))
    log.write('\n')



    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')


    log.write('\ttsplit   = %s\n'%(split))
    log.write('\tlen(ids) = %d\n'%(len(ids)))
    log.write('\n')


    for tag_name, do_test_augment, undo_test_augment, params in augments:

        ## setup  --------------------------
        tag = 'xx_gray_%s'%tag_name   ##tag = 'test1_ids_gray2_53-00011000_model'
        os.makedirs(out_dir +'/predict/%s/overlays'%tag, exist_ok=True)
        os.makedirs(out_dir +'/predict/%s/predicts'%tag, exist_ok=True)

        os.makedirs(out_dir +'/predict/%s/rcnn_proposals'%tag, exist_ok=True)
        os.makedirs(out_dir +'/predict/%s/detections'%tag, exist_ok=True)
        os.makedirs(out_dir +'/predict/%s/masks'%tag, exist_ok=True)
        os.makedirs(out_dir +'/predict/%s/instances'%tag, exist_ok=True)



        log.write('** start evaluation here @%s! **\n'%tag)
        for i in range(0,len(ids)):


            folder, name = ids[i].split('/')[-2:]
            print('%03d %s'%(i,name))

            ##name ='33d6d8e9d74f9da9679000a6cf551fffe4ad45af7d9679e199c5c4bd2d1e0741'
            ##name='4727d94c6a57ed484270fdd8bbc6e3d5f2f15d5476794a4e37a40f2309a091e2' #debug errorimages
            #name='0ed3555a4bd48046d3b63d8baf03a5aa97e523aa483aaa07459e7afa39fb96c6'
            ##name='0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac'

            image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png'%(folder,name), cv2.IMREAD_COLOR)
            augment_image  = do_test_augment(image,  **params)


            net.set_mode('test')
            with torch.no_grad():
                input = torch.from_numpy(augment_image.transpose((2,0,1))).float().div(255).unsqueeze(0)
                input = Variable(input).cuda()
                net.forward(input)


            mask = undo_test_augment_identity(net, image)
            contour_overlay = mask_to_contour_overlay(mask, do_gamma(image,2.5), (0,255,0))


            #save
            np.save(out_dir + '/predict/%s/masks/%s.npy' % (tag, name), mask)
            cv2.imwrite(out_dir +'/predict/%s/overlays/%s.png'%(tag,name),contour_overlay)


            image_show('image', image)
            image_show('contour_overlay', contour_overlay)
            cv2.waitKey(1)
            continue




        #assert(test_num == len(test_loader.sampler))
        log.write('-------------\n')
        log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
        log.write('tag=%s\n'%tag)
        log.write('\n')



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_predict()

    print('\nsucess!')
