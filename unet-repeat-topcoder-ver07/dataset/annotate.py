from common import *
from utility.file import *
from utility.draw import *

from dataset.reader import *






def apply_fix_annotation():

    # png_dir   = '/root/share/project/kaggle/science2018/data/image/ext-BBBC006/fixes'
    # image_dir = '/root/share/project/kaggle/science2018/data/image/ext-BBBC006/images'
    # data_dir  = '/root/share/project/kaggle/science2018/data/image/ext-BBBC006'

    # png_dir   = '/root/share/project/kaggle/science2018/data/image/ext-he-ellen/fixes'
    # image_dir = '/root/share/project/kaggle/science2018/data/image/ext-he-ellen/images'
    # data_dir  = '/root/share/project/kaggle/science2018/data/image/ext-he-ellen'

    png_dir   = '/root/share/project/kaggle/science2018/data/image/ext-he-b/fixes'
    image_dir = '/root/share/project/kaggle/science2018/data/image/ext-he-b/images'
    data_dir  = '/root/share/project/kaggle/science2018/data/image/ext-he-b'

    ## start -----------------------------
    os.makedirs(data_dir + '/masks', exist_ok=True)
    os.makedirs(data_dir + '/overlays', exist_ok=True)
    os.makedirs(data_dir + '/images', exist_ok=True)
    os.makedirs(data_dir + '/psds', exist_ok=True)


    png_files = glob.glob(png_dir + '/*.png')
    #png_files = [f.replace('.mask.png','.fix.png') for f in png_files]
    #png_files = [f.replace('.png','.fix.png') for f in png_files]

    for png_file in png_files:
        #name = png_file.split('/')[-1].replace('.fix','').replace('.png','')
        name = png_file.split('/')[-1].replace('.fix','').replace('.png','')
        print(name)


        image_file = data_dir +'/images/%s.png'%name
        mask  = color_overlay_to_mask(cv2.imread(png_file,cv2.IMREAD_COLOR))
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)
        norm_image = do_gamma(image, 2.5)


        #------------------------------------
        #save
        np.save(data_dir + '/masks/%s.npy' % name, mask)
        cv2.imwrite(data_dir + '/images/%s.png' % name, image)


        #check
        color_overlay    = mask_to_color_overlay  (mask,color='summer')
        color_overlay1   = mask_to_contour_overlay(mask,color_overlay,[255,255,255])
        contour_overlay  = mask_to_contour_overlay(mask,image,[0,255,0])
        contour_overlay1 = mask_to_contour_overlay(mask,norm_image,[0,255,0])
        all = np.hstack((image, contour_overlay1,color_overlay1,)).astype(np.uint8)


        #save for photoshop
        cv2.imwrite(data_dir + '/overlays/%s.png' % name, all)

        #psd
        os.makedirs(data_dir +'/overlays/%s'%(name), exist_ok=True)
        cv2.imwrite(data_dir +'/overlays/%s/%s.png'%(name,name),image)
        cv2.imwrite(data_dir +'/overlays/%s/%s.norm.png'%(name,name),norm_image)
        cv2.imwrite(data_dir +'/overlays/%s/%s.mask.png'%(name,name),color_overlay)
        cv2.imwrite(data_dir +'/overlays/%s/%s.contour.png'%(name,name),contour_overlay)


        image_show('all', all)
        cv2.waitKey(1)


def run_make_train_annotation():

    split = 'ext_he_a'  #'train1_ids_all_670'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

    data_dir = DATA_DIR  + '/image/ext-he-a'
    os.makedirs(data_dir + '/masks', exist_ok=True)
    os.makedirs(data_dir + '/overlays', exist_ok=True)
    os.makedirs(data_dir + '/images', exist_ok=True)


    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]
        folder, name   = id.split('/')[-2:]
        print(name)


        #image
        image_file = DATA_DIR + '/image/%s/images/%s.png'%(folder,name)
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)
        norm_image = do_gamma(image, 2.5)

        H,W,C      = image.shape
        mask = np.zeros((H,W), np.int32)

        mask_files = glob.glob(DATA_DIR + '/image/%s/__download__/extra_data/%s/masks/*.png'%(folder,name))
        mask_files.sort()
        num_masks = len(mask_files)
        for i in range(num_masks):
            mask_file = mask_files[i]
            m = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
            mask[np.where(m>128)] = i+1

        #save
        np.save(data_dir + '/masks/%s.npy' % name, mask)
        #cv2.imwrite(data_dir + '/images/%s.png' % name, image)


        #check --------------------------------------------------------------------------
        color_overlay    = mask_to_color_overlay  (mask,color='summer')
        color_overlay1   = mask_to_contour_overlay(mask,color_overlay,[255,255,255])
        contour_overlay  = mask_to_contour_overlay(mask,image,[0,255,0])
        contour_overlay1 = mask_to_contour_overlay(mask,norm_image,[0,255,0])
        all = np.hstack((image, contour_overlay1,color_overlay1,)).astype(np.uint8)


        #save for photoshop
        cv2.imwrite(data_dir + '/overlays/%s.png' % name, all)

        #psd
        os.makedirs(data_dir +'/overlays/%s'%(name), exist_ok=True)
        cv2.imwrite(data_dir +'/overlays/%s/%s.png'%(name,name),image)
        cv2.imwrite(data_dir +'/overlays/%s/%s.norm.png'%(name,name),norm_image)
        cv2.imwrite(data_dir +'/overlays/%s/%s.mask.png'%(name,name),color_overlay)
        cv2.imwrite(data_dir +'/overlays/%s/%s.contour.png'%(name,name),contour_overlay)


        image_show('all', all)
        cv2.waitKey(1)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #apply_fix_annotation()
    run_make_train_annotation()

    print('\nsucess!')
