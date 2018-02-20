import pandas as pd

from load_data import load_train_data, load_test_data, load_test_image_sizes
from model import model
from metrics import my_iou_metric, dice_coef_loss
from process_data import upsample, encode
from submit import submit

test_path = '.kaggle/competitions/data-science-bowl-2018/test/'
test_ids = next(os.walk(test_path))[1]

# if __name__ == '__main__':

def main():
    print('Loading Data')

    x_train, y_train = load_train_data()
    x_test = load_test_data()
    x_test_sizes = load_test_image_sizes()

    print('Making model')
    model = model()

    print('Fitting model')
    model.fit(x_train, y_train, batch_size=8, validation_split=0.1, epochs=10)

    print('Predict')
    preds = model.predict(x_test, verbose=1)
    preds = (preds > 0.5).astype(np.uint8)

    print('Upsample')
    upsampled = upsample(preds, x_test_sizes)

    print('Encode predictions')
    rles, ids = encode(upsampled, test_ids)

    print('Submit')
    submit(rles, ids)

    pd.read_csv('./submission.csv')