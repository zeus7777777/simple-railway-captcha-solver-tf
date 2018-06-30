import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parse_csv_captcha(filename):
    files = []
    captchas = []
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')
        for line in lines:
            line = line.strip().split(',')
            files.append(line[0])
            captchas.append(line[1])
    return files, captchas

def captcha_to_int(captcha):
    ans = []
    for i in range(len(captcha)):
        c = ord(captcha[i])
        if 48<=c<58:
            ans.append(c-48)
        elif 65<=c<=90:
            ans.append(c-55)
        else:
            assert(False)
    if len(captcha)==5:
        ans.append(36)
    assert(len(ans)==6)
    return ans

def main(captcha_file, prefix, tfrecord_path):
    all_files = []
    all_lens = []
    all_captcha = []

    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    cnt = 0

    file_names, captchas = parse_csv_captcha(captcha_file)
    for j in range(len(file_names)):
        with tf.gfile.FastGFile(prefix+file_names[j]+'.jpg' , 'rb') as f:
            jpeg_str = f.read()
        example = tf.train.Example(features=tf.train.Features(feature={
            'jpeg_str': _bytes_feature(tf.compat.as_bytes(jpeg_str)),
            'captcha_ints': _int64_feature(captcha_to_int(captchas[j])),
        }))
        writer.write(example.SerializeToString())
        cnt += 1
        if (cnt+1)%1000==0:
            print(cnt+1, 'images processed')
        if cnt<100:
            print(file_names[j], captchas[j])
    writer.close()
    print(cnt, 'images written')

if __name__=='__main__':
    captcha_files = [
        'data/5_imitate_train_set/captcha_train.csv',
        'data/5_imitate_vali_set/captcha_vali.csv',
        'data/6_imitate_train_set/captcha_train.csv',
        'data/6_imitate_vali_set/captcha_vali.csv',
        'data/56_imitate_train_set/captcha_train.csv',
        'data/56_imitate_vali_set/captcha_vali.csv'
    ]
    prefix = [
        'data/5_imitate_train_set/',
        'data/5_imitate_vali_set/',
        'data/6_imitate_train_set/',
        'data/6_imitate_vali_set/',
        'data/56_imitate_train_set/',
        'data/56_imitate_vali_set/'
    ]
    tfrecord_path = [
        'tfrecord/images_train_5.tfrecord',
        'tfrecord/images_valid_5.tfrecord',
        'tfrecord/images_train_6.tfrecord',
        'tfrecord/images_valid_6.tfrecord',
        'tfrecord/images_train_56.tfrecord',
        'tfrecord/images_valid_56.tfrecord'
    ]
    for i in range(len(captcha_files)):
        main(captcha_files[i], prefix[i], tfrecord_path[i])
