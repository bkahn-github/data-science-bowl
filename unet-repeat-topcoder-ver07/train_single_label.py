import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'  #'3,2' #'3,2,1,0'

from common import *
from utility.file   import *
from dataset.reader import *
from net.rate   import *
from net.metric import *

#from draw import *

# -------------------------------------------------------------------------------------

#from net.se_resnext50_unet_single_label.model import *
from net.se_resnext101_unet_single_label.model import *
Net = UNet


#WIDTH, HEIGHT = 128,128
#WIDTH, HEIGHT = 192,192
WIDTH, HEIGHT = 256,256
# -------------------------------------------------------------------------------------

def mask_to_annotation(mask):
    foreground = mask_to_foreground(mask)
    cut = mask_to_cut(mask)
    label = foreground.astype(np.int32)
    label[np.where(cut)]=2
    return  label



def train_augment(image, mask, index):

    # illumintaion ------------
    if 0:
        type = random.randint(0,4)
        if type==0:
            image = random_transform(image, u=0.5, func=do_custom_process1, gamma=[0.8,2.5],alpha=[0.7,0.9],beta=[1.0,2.0])

        elif type==1:
            image = random_transform(image, u=0.5, func=do_contrast, alpha=[0.5,2.5])

        elif type==2:
            image = random_transform(image, u=0.5, func=do_gamma, gamma=[1,3])

        elif type==3:
            image = random_transform(image, u=0.5, func=do_clahe, clip=[1,3], grid=[8,16])

        else:
            pass
        #print('illumintaion',image.dtype)

    # filter/noise ------------
    if 0:
        type = random.randint(0,2)
        if type==0:
            image = random_transform(image, u=0.5, func=do_unsharp, size=[9,19], strength=[0.2,0.4], alpha=[4,6])

        elif type==1:
            image = random_transform(image, u=0.5, func=do_speckle_noise, sigma=[0.1,0.5])

        else:
            pass
        #print('filter',image.dtype)

    # geometric ------------
    if 1:
        # type = random.randint(0,2)
        # if type==0:
        #     image, mask = random_transform2(image, mask, u=0.5, func=do_stretch2, scale_x=[1,2], scale_y=[1,1] )
        # if type==1:
        #     image, mask = random_transform2(image, mask, u=0.5, func=do_stretch2, scale_x=[1,1], scale_y=[1,2] )
        #
        #
        # image, mask = random_transform2(image, mask, u=0.5, func=do_shift_scale_rotate2, dx=[0,0],dy=[0,0], scale=[1/2,2], angle=[-45,45])
        # image, mask = random_transform2(image, mask, u=0.5, func=do_shift_scale_rotate2, dx=[0,0],dy=[0,0], scale=[1/2,2], angle=[-45,45])
        # image, mask = random_transform2(image, mask, u=0.5, func=do_elastic_transform2, grid=[8,64], distort=[0,0.5])

        image, mask = random_crop_transform2(image, mask, WIDTH, HEIGHT, u=0.5)
        image, mask = do_flip_transpose2(image, mask, random.randint(0,8))
        #print('geometric',image.dtype)


    #---------------------------------------
    #image, mask = fix_crop_transform2(image, mask, -1,-1,WIDTH, HEIGHT)

    input = torch.from_numpy(image.transpose((2,0,1))).float().div(255)
    label = mask_to_annotation(mask)
    label = torch.from_numpy(label[np.newaxis,...]).long()

    return input, label, image, mask, index




def valid_augment(image, mask, index):

    image,  mask = fix_crop_transform2(image, mask, -1,-1,WIDTH, HEIGHT)

    #---------------------------------------
    input = torch.from_numpy(image.transpose((2,0,1))).float().div(255)
    label = mask_to_annotation(mask)
    label = torch.from_numpy(label[np.newaxis,...]).long()

    return input, label, image, mask, index



def train_collate(batch):
    batch_size = len(batch)
    inputs     = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    labels     = torch.stack([batch[b][1]for b in range(batch_size)], 0)
    images     =             [batch[b][2]for b in range(batch_size)]
    masks      =             [batch[b][3]for b in range(batch_size)]
    indices    =             [batch[b][4]for b in range(batch_size)]

    return [inputs, labels, images, masks, indices]




### training ##############################################################
def evaluate( net, test_loader ):

    test_num  = 0
    test_loss = np.zeros(6,np.float32)
    #return test_loss

    for i, (inputs, labels_truth, images, masks_truth, indices) in enumerate(test_loader, 0):

        with torch.no_grad():
            inputs       = Variable(inputs).cuda()
            labels_truth = Variable(labels_truth).cuda()

            net.forward( inputs )
            net.criterion( labels_truth )

        batch_size = len(indices)
        test_loss += batch_size*np.array((
                           net.loss.cpu().data.numpy(),
                           0,
                           0,
                           0,
                           0,
                           0,
                         ))
        test_num  += batch_size

    assert(test_num == len(test_loader.sampler))
    test_loss = test_loss/test_num
    return test_loss



#--------------------------------------------------------------
def run_train():

    out_dir = RESULTS_DIR + '/unet-se-resnext101-single-9a'
    initial_checkpoint = \
        RESULTS_DIR + '/unet-se-resnext101-single-9a/checkpoint/00011400_model.pth'
        #

    pretrain_file = \
        None  #RESULTS_DIR + '/unet-se-resnext50-00/checkpoint/00004400_model.pth'
    skip = ['logit','foreground'] #['crop','mask']

    ## setup  -----------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/train', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')



    ## net ----------------------
    log.write('** net setting **\n')
    cfg = Configuration()
    net = Net(cfg).cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        # cfg = load_pickle_file(out_dir +'/checkpoint/configuration.pkl')

    if pretrain_file is not None:
        log.write('\tpretrain_file = %s\n' % pretrain_file)
        net.load_pretrain( pretrain_file, skip)

    log.write('%s\n\n'%(type(net)))
    log.write('%s\n'%(net.version))
    log.write('\n')



    ## optimiser ----------------------------------
    iter_accum  = 1
    batch_size  = 8

    num_iters   = 1000  *1000
    iter_smooth = 20
    iter_log    = 50
    iter_valid  = 100
    iter_save   = [0, num_iters-1]\
                   + list(range(0,num_iters,200))#1*1000


    LR = None  #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.005/iter_accum, momentum=0.9, weight_decay=0.0001)

    start_iter = 0
    start_epoch= 0.
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']

        rate = get_learning_rate(optimizer)  #load all except learning rate
        optimizer.load_state_dict(checkpoint['optimizer'])
        adjust_learning_rate(optimizer, rate)
        pass


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = ScienceDataset(
                            #'valid1_ids_gray2_38',
                            'train1_ids_gray2_500',
                            #'debug1_ids_gray_only_10',
                            #'disk0_ids_dummy_9',
                            #'train1_ids_purple2_80',
                            #'merge1_1',
                            mode='train',transform = train_augment)

    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = train_collate)


    valid_dataset = ScienceDataset(
                            'valid1_ids_gray2_38',
                            #'valid1_ids_gray2_43',
                            #'debug1_ids_gray_only_10',
                            #'disk0_ids_dummy_9',
                            #'valid1_ids_purple2_20',
                            #'merge1_1',
                            mode='train',transform = valid_augment)

    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = train_collate)

    log.write('\tWIDTH, HEIGHT = %d, %d\n'%(WIDTH, HEIGHT))
    log.write('\ttrain_dataset.split = %s\n'%(train_dataset.split))
    log.write('\tvalid_dataset.split = %s\n'%(valid_dataset.split))
    log.write('\tlen(train_dataset)  = %d\n'%(len(train_dataset)))
    log.write('\tlen(valid_dataset)  = %d\n'%(len(valid_dataset)))
    log.write('\tlen(train_loader)   = %d\n'%(len(train_loader)))
    log.write('\tlen(valid_loader)   = %d\n'%(len(valid_loader)))
    log.write('\tbatch_size  = %d\n'%(batch_size))
    log.write('\titer_accum  = %d\n'%(iter_accum))
    log.write('\tbatch_size*iter_accum  = %d\n'%(batch_size*iter_accum))
    log.write('\n')

    #<debug>========================================================================================
    if 0:
        for inputs, labels_truth, images, masks_truth, indices in train_loader:

            batch_size, C,H,W = inputs.size()
            print('batch_size=%d'%batch_size)

            images = inputs.cpu().numpy()
            labels_truth = labels_truth.data.cpu().numpy()

            for b in range(batch_size):
                image = (images[b].transpose((1,2,0))*255)
                mask_truth  = masks_truth[b]
                label_truth = labels_truth[b]

                contour_overlay = mask_to_contour_overlay(mask_truth,  image, [0,255,0])
                image_show('contour_overlay',contour_overlay,resize=1)
                image_show_norm('label_truth',label_truth,resize=1)

                cv2.waitKey(0)
    #<debug>========================================================================================


    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' momentum=%f\n'% optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n'%str(LR) )

    log.write(' images_per_epoch = %d\n\n'%len(train_dataset))
    log.write(' rate    iter   epoch  num   | valid_loss               | train_loss               | batch_loss               |  time          \n')
    log.write('-------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss  = np.zeros(6,np.float32)
    valid_loss  = np.zeros(6,np.float32)
    batch_loss  = np.zeros(6,np.float32)
    rate = 0

    start = timer()
    j = 0
    i = 0


    while  i<num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6,np.float32)
        sum = 0

        net.set_mode('train')
        optimizer.zero_grad()
        for inputs, labels_truth, images, masks_truth, indices in train_loader:
            #if all(len(b)==0 for b in truth_boxes): continue

            batch_size = len(indices)
            i = j/iter_accum + start_iter
            epoch = (i-start_iter)*batch_size*iter_accum/len(train_dataset) + start_epoch
            num_products = epoch*len(train_dataset)

            if i % iter_valid==0:
                net.set_mode('valid')
                valid_loss = evaluate(net, valid_loader)
                net.set_mode('train')

                print('\r',end='',flush=True)
                log.write('%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f %0.2f | %0.3f   %0.2f %0.2f %0.2f | %0.3f   %0.2f %0.2f %0.2f | %s\n' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], #valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], #train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], #batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)

            #if 1:
            if i in iter_save:
                torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(i))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : i,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(i))
                save_pickle_file(out_dir +'/checkpoint/configuration.pkl', cfg)

            # learning rate schduler -------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr/iter_accum)
            rate = get_learning_rate(optimizer)*iter_accum


            # one iteration update  -------------
            inputs       = Variable(inputs).cuda()
            labels_truth = Variable(labels_truth).cuda()

            net.forward( inputs )
            net.criterion( labels_truth )

            # accumulated update
            net.loss.backward()
            if j%iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()


            # print statistics  ------------
            batch_loss = np.array((
                           net.loss.cpu().data.numpy(),
                           0,
                           0,
                           0,
                           0,
                           0,
                         ))
            sum_train_loss += batch_loss
            sum += 1
            if i%iter_smooth == 0:
                train_loss = sum_train_loss/sum
                sum_train_loss = np.zeros(6,np.float32)
                sum = 0


            print('\r%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f %0.2f | %0.3f   %0.2f %0.2f %0.2f | %0.3f   %0.2f %0.2f %0.2f | %s  %d,%d,%s' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], #valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], #train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], #batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60) ,i,j, ''), end='',flush=True)#str(inputs.size()))
            j=j+1

            #<debug> ===================================================================
            if 1:
            #if i%10==0:

                net.set_mode('test')
                with torch.no_grad():
                    net.forward( inputs )

                batch_size,C,H,W = inputs.size()
                logits = net.logits.data.cpu().numpy()
                probs  = np_softmax(logits)
                labels = np.argmax(probs,1)

                labels_truth = labels_truth.data.cpu().numpy()
                num_classes = 3

                #print('train',batch_size)
                for b in range(batch_size):
                    image       = images[b]
                    mask_truth  = masks_truth[b]
                    label_truth = labels_truth[b]

                    prob  = probs[b]
                    label = labels[b]

                    contour_overlay = do_gamma(image,2.5)
                    ## post processing
                    if 0:
                        binary = label>0
                        marker = skimage.morphology.label(label==1)
                        distance = -label

                        water = skimage.morphology.watershed(-distance, marker, connectivity=1, mask=binary)
                        contour_overlay = mask_to_contour_overlay(water, do_gamma(image,2.5), (0,255,0))

                        image_show('water', contour_overlay,1)
                        #image_show_norm('distance', distance,resize=1)

                    ## draw --------------------------------------------------------------------------
                    #contour_overlay = multi_mask_to_contour_overlay(truth_mask, image, [255,255,0] )
                    #color_overlay   = multi_mask_to_color_overlay(mask)

                    #contour_overlay  = mask_to_contour_overlay(mask_truth, norm_image, (0,255,0))
                    #contour_overlay1 = mask_to_contour_overlay(water, norm_image, (0,255,0))

                    #norm_image = norm_image.copy()
                    #norm_image[:,:,2] = (1-(1-norm_image[:,:,2]/255)*(1-cut))*255


                    overlay_label_truth = np.zeros((H,W,3), np.uint8)
                    overlay_label_truth[:,:,0] = (label_truth==1)*255
                    overlay_label_truth[:,:,1] = (label_truth==2)*255

                    overlay_label = np.zeros((H,W,3), np.uint8)
                    overlay_label[:,:,0]= prob[1,:,:]*255
                    overlay_label[:,:,1]= prob[2,:,:]*255

                    all = np.vstack([
                        np.hstack([overlay_label_truth, contour_overlay ]),
                        np.hstack([overlay_label, contour_overlay]),
                    ])

                    image_show('all', all,1)
                    #image_show_norm('truth_border', truth_border,1)
                    # image_show_norm('border', border,1)
                    # image_show_norm('truth_foreground', truth_foreground,1)
                    # image_show_norm('foreground', foreground,1)


                    name = train_dataset.ids[indices[b]].split('/')[-1]
                    cv2.imwrite(out_dir +'/train/%s.png'%name,all)
                    cv2.waitKey(1)
                    pass


                net.set_mode('train')
            #<debug> ===================================================================


        pass  #-- end of one data loader --
    pass #-- end of all iterations --


    if 1: #save last
        torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

    log.write('\n')



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()

    print('\nsucess!')



#  ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#
#
