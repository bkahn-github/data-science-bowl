from common import *
from net.metric import *
from dataset.reader import *


def csv_to_masks(csv_file, split):

    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#') #[2507:]
    sorted(ids)

    df  = pd.read_csv(csv_file, sep=',')
    ## image_ids = df['ImageId'].unique()

    masks = []
    for i, id in enumerate(ids):
        folder, name  = id.split('/')
        print('%05d : name = %s'%(i,name))

        image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png'%(folder,name), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]

        mask = np.zeros((height,width),np.int32)
        rles = df.loc[df['ImageId'] == name]['EncodedPixels'].values
        for t, rle in enumerate(rles):
            binary = run_length_decode(rle, height, width, fill_value=255)
            mask[binary>0]=t+1

        masks.append(mask)

        if 1: #debug
            image_show('image', image, 1)
            image_show('mask',  mask_to_color_overlay(mask), 1)
            #image_show('binary',  binary, 1)
            cv2.waitKey(1)




def run_stage1_test_server_evaluation():

    split = 'test1_ids_all_65'
    #csv_file = '/root/share/project/kaggle/science2018/data/stage1_solution.csv'
    #csv_file = '/root/share/project/kaggle/science2018/results/__submit__/LB-0.523/submission-add-71-0.569b.csv'
    #csv_file = '/root/share/project/kaggle/science2018/results/__submit__/LB-0.523/submission-aaa-0.417.csv'
    #csv_file = '/root/share/project/kaggle/science2018/results/__submit__/LB-0.523/submission-add-74-0.570a.csv'
    csv_file = '/root/share/project/kaggle/science2018/results/unet-se-resnext50-7/submit/submission-gray.csv'

    #------------------------------------------------------------------------------

    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#') #[2507:]
    sorted(ids)

    df = pd.read_csv(csv_file, sep=',')
    if 1:  #check if all image ids are included ------------------
        image_ids = df['ImageId'].unique()
        sorted(image_ids)
        for i, id in enumerate(ids):
            folder, name  = id.split('/')
            if name in image_ids: continue
            print ('missing %s in csv file',name)
            exit(0)
    #--------------------------------------------------------------


    mask_average_precisions = []
    for i, id in enumerate(ids):
        folder, name  = id.split('/')

        #image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png'%(folder,name), cv2.IMREAD_COLOR)
        #height, width = image.shape[:2]

        truth_mask = np.load(DATA_DIR + '/image/%s/masks/%s.npy'%(folder,name))
        height,width = truth_mask.shape[:2]

        mask = np.zeros((height,width),np.int32)
        rles = df.loc[df['ImageId'] == name]['EncodedPixels'].values
        for t, rle in enumerate(rles):
            if type(rle)==float:
                if math.isnan(rle): continue
            #print(type(rle),rle)
            binary = run_length_decode(rle, height, width, fill_value=255)
            mask[binary>0]=t+1

        #-----------------------------------------------
        mask_average_precision, mask_precision =\
                compute_average_precision_for_mask(mask, truth_mask, t_range=np.arange(0.5, 1.0, 0.05))

        mask_average_precisions.append(mask_average_precision)
        print('%05d : name = %s   %0.5f'%(i,name,mask_average_precision))

    mask_average_precisions = np.array(mask_average_precisions)
    print('LB score for stage1 test: ', mask_average_precisions.mean(0))




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # csv_file = '/root/share/project/kaggle/science2018/data/stage1_solution.csv'
    # split    = 'test1_ids_all_65'
    # masks = csv_to_masks(csv_file, split)
    run_stage1_test_server_evaluation()

    print('\nsucess!')
