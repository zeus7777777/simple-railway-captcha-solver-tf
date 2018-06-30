# simple-railway-captcha-solver-tf
Tensorflow re-implement of https://github.com/JasonLiTW/simple-railway-captcha-solver (Because original repository's code need large memory to train).
It is recommended to read original repository's README.

## Requirement
 - Tensorflow >= 1.6
 - PIL
 - Selenium


## Online Demo
```
python3 demo_online.py
```

## Training

### Create training/validation set.
```
python3 captcha_gen.py
```
This may take a long time.

### Create tfrecord file.
```
python3 create_tfrecord.py
```

### Training
```
python3 train5.py
python3 train6.py
python3 train56.py
```
