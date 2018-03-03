import pandas as pd

def submit(data, ids):
    sub = pd.DataFrame()
    sub['ImageId'] = ids
    sub['EncodedPixels'] = pd.Series(data).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('submission.csv', index=False)