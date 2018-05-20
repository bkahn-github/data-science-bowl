import os, sys
sys.path.append(os.path.dirname(__file__))

#--------------------------------------------------------------

from train_mask_rcnn_net_3 import * #se_resnext50_mask_rcnn_2crop
#from train_mask_rcnn_net_2 import * #se_resnext101_mask_rcnn


#--------------------------------------------------------------
from predict_argumentation import *



def run_predict():

    #out_dir = RESULTS_DIR + '/mask-se-resnext50-rcnn_2crop-mega-05d'
    out_dir = RESULTS_DIR + '/stage_2'
    initial_checkpoint = \
       RESULTS_DIR + '/mask-se-resnext50-rcnn_2crop-mega-05d/checkpoint/00057500_model.pth'

    # augment -----------------------------------------------------------------------------------------------------
    augments=[
        #('normal',           do_test_augment_identity,       undo_test_augment_identity,       {         } ),
        #('flip_transpose_1', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':1,} ),

        #('flip_transpose_2', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':2,} ), ## gpu 1
        #('flip_transpose_3', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':3,} ),
        #('flip_transpose_4', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':4,} ),

        #('flip_transpose_5', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':5,} ), ## gpu 2
        #('flip_transpose_6', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':6,} ),
        #('flip_transpose_7', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':7,} ),

        #('scale_0.8',        do_test_augment_scale,  undo_test_augment_scale,     { 'scale_x': 0.8, 'scale_y': 0.8  } ), ## gpu 3
        #('scale_1.2',        do_test_augment_scale,  undo_test_augment_scale,     { 'scale_x': 1.2, 'scale_y': 1.2  } ),
        #('scale_0.5',        do_test_augment_scale,  undo_test_augment_scale,     { 'scale_x': 0.5, 'scale_y': 0.5  } ),
        ('scale_1.8',        do_test_augment_scale,  undo_test_augment_scale,     { 'scale_x': 1.8, 'scale_y': 1.8  } ),

        # not used -------------------------
        # ('scale_0.8_flip_transpose_1',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':1,  } ),
        # ('scale_0.8_flip_transpose_2',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':2,  } ),
        # ('scale_0.8_flip_transpose_3',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':3,  } ),
        # ('scale_0.8_flip_transpose_4',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':4,  } ),
        # ('scale_0.8_flip_transpose_5',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':5,  } ),
        # ('scale_0.8_flip_transpose_6',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':6,  } ),
        # ('scale_0.8_flip_transpose_7',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.8, 'scale_y': 0.8,  'type':7,  } ),
        #
        # ('scale_1.2_flip_transpose_1',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':1,  } ),
        # ('scale_1.2_flip_transpose_2',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':2,  } ),
        # ('scale_1.2_flip_transpose_3',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':3,  } ),
        # ('scale_1.2_flip_transpose_4',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':4,  } ),
        # ('scale_1.2_flip_transpose_5',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':5,  } ),
        # ('scale_1.2_flip_transpose_6',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':6,  } ),
        # ('scale_1.2_flip_transpose_7',        do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 1.2, 'scale_y': 1.2,  'type':7,  } ),
        #
        # ('stretch_0.5_flip_transpose_1',      do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.5, 'scale_y': 1.0,  'type':1,  } ),
        # ('stretch_0.5_flip_transpose_2',      do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.5, 'scale_y': 1.0,  'type':2,  } ),
        # ('stretch_0.5_flip_transpose_3',      do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.5, 'scale_y': 1.0,  'type':3,  } ),
        # ('stretch_0.5_flip_transpose_4',      do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.5, 'scale_y': 1.0,  'type':4,  } ),
        # ('stretch_0.5_flip_transpose_5',      do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.5, 'scale_y': 1.0,  'type':5,  } ),
        # ('stretch_0.5_flip_transpose_6',      do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.5, 'scale_y': 1.0,  'type':6,  } ),
        # ('stretch_0.5_flip_transpose_7',      do_test_augment_scale_flip_transpose,  undo_test_augment_scale_flip_transpose,     { 'scale_x': 0.5, 'scale_y': 1.0,  'type':7,  } ),

    ]

    #-----------------------------------------------------------------------------------

    split = 'test2_ids_gray_2800'  #'BBBC006'   #'valid1_ids_gray2_43'
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
    cfg.rcnn_test_nms_pre_score_threshold = 0.5
    cfg.mask_test_nms_pre_score_threshold = cfg.rcnn_test_nms_pre_score_threshold

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
        for i in range(1565,len(ids)):

            #error images for x1.8
            if i in [1445,1466, 1502, 1565]: continue  #1565

            folder, name = ids[i].split('/')[-2:]
            print('%03d %s'%(i,name))

            #name='4727d94c6a57ed484270fdd8bbc6e3d5f2f15d5476794a4e37a40f2309a091e2' #debug errorimages
            #name='0ed3555a4bd48046d3b63d8baf03a5aa97e523aa483aaa07459e7afa39fb96c6'
            #name='0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac'

            image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png'%(folder,name), cv2.IMREAD_COLOR)
            augment_image  = do_test_augment(image, proposal=None,  **params)

            try:
                net.set_mode('test')
                with torch.no_grad():
                    input = torch.from_numpy(augment_image.transpose((2,0,1))).float().div(255).unsqueeze(0)
                    input = Variable(input).cuda()
                    net.forward(input)

            except Exception:
                print (' error',end='\n',flush=True)
                exit(0)


            if len(net.detections) ==0:

                # dummy zero results
                height,width  = image.shape[:2]
                rcnn_proposal = np.zeros((0,8), np.float32)
                detection     = np.zeros((0,8), np.float32)
                mask          = np.zeros((height,width), np.int32)
                instance      = np.zeros((0, height,width), np.uint8)

                # visual results
                empty = np.zeros((height,width,3),np.uint8)
                all  = np.hstack((image, empty, empty, empty))
                all2 = np.hstack([image, empty, empty, empty])
                #continue
            else:


                rcnn_proposal, detection, mask, instance = undo_test_augment(net, image, **params)

                # visual results
                norm_image = do_gamma(image,2.5)
                threshold = 0.8
                all2 = draw_predict_mask(threshold, image, mask, detection)

                #norm_image      = do_gamma(image,2.5)
                color_overlay   = mask_to_color_overlay(mask)
                color1_overlay  = mask_to_contour_overlay(mask, color_overlay)
                contour_overlay = mask_to_contour_overlay(mask, norm_image, [0,255,0])
                mask_score = instance.sum(0)
                mask_score = cv2.cvtColor((mask_score/mask_score.max()*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
                all = np.hstack((image, contour_overlay, color1_overlay, mask_score)).astype(np.uint8)


            #save and show here ----------------------------------------------------------------------------
            ##np.save(out_dir +'/predict/%s/rcnn_proposals/%s.npy'%(tag,name),rcnn_proposal)
            np.save(out_dir + '/predict/%s/masks/%s.npy' % (tag, name), mask)
            np.save(out_dir + '/predict/%s/detections/%s.npy' % (tag, name), detection)
            np.save(out_dir + '/predict/%s/instances/%s.npy' % (tag, name), (instance * 255).astype(np.uint8))

            #psd
            os.makedirs(out_dir +'/predict/overlays', exist_ok=True)
            cv2.imwrite(out_dir +'/predict/%s/overlays/%s.png'%(tag,name),all)
            cv2.imwrite(out_dir +'/predict/%s/predicts/%s.png'%(tag,name), all2)

            image_show('overlays', all)
            image_show('predict_mask', all2)
            cv2.waitKey(1)
            # save and show here ----------------------------------------------------------------------------


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
