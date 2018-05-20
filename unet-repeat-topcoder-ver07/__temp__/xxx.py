from common import *

from dataset_file import*
from net_file import*

def make_fix_length(wave,length=16000):

    if len(wave)>length:
        wave = wave[:length]
    else:
        wave = np.pad(wave, (0,length-len(wave)), 'constant', constant_values=(0, 0))
    return wave

def wave_to_spect(wave):

    spect = ...

    return spect

#-------------------------------------------------------------------



def run_predict():


    dataset = ScienceDataset(
                'folder_name',
                 ...)


    net = ...


    num_sampers = len(dataset)
    for n in range(num_sampers):
        wave, label = dataset[n]

        #------------------------------------
        #post-process here
        augmented_wave = make_fix_length(wave)
        spect = make_log_mel ....(augmented_wave)
        #------------------------------------

        prob = net(spect)
        print(prob)





    pass


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_predict()

    print('\nsucess!')
