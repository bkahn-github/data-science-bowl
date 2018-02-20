# data-science-bowl

## Installation

### Install packages
```pip install -r data-science-bowl/requirements.txt``` or ```pip3 install -r requirements.txt```

### Create a .kaggle directory
```!mkdir ~/.kaggle```

### Create a `kaggle.json` file
```python
f = open(".kaggle/kaggle.json", "w")
f.write('{"username":"bkkaggle","key":[KEY]}')
f.close()
```

### Download the data 
`!kaggle competitions download -c data-science-bowl-2018`

### unzip the data
```
!unzip ~/.kaggle/competitions/data-science-bowl-2018/stage1_train.zip -d ~/.kaggle/competitions/data-science-bowl-2018/train
!unzip ~/.kaggle/competitions/data-science-bowl-2018/stage1_test.zip -d ~/.kaggle/competitions/data-science-bowl-2018/test
!unzip ~/.kaggle/competitions/data-science-bowl-2018/stage1_train_labels.csv.zip -d ~/.kaggle/competitions/data-science-bowl-2018/labels
```
