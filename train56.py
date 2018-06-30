import time
import os

import tensorflow as tf

import model56 as model
from config import *

def main():
    train_graph = tf.Graph()
    valid_graph = tf.Graph()

    with train_graph.as_default():
        train_model = model.TrainModel(batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, tfrecord_file='tfrecord/images_train_56.tfrecord')
    with valid_graph.as_default():
        valid_model = model.ValidModel(batch_size=BATCH_SIZE, tfrecord_file='tfrecord/images_valid_56.tfrecord')
    
    train_sess = tf.Session(graph=train_graph)
    valid_sess = tf.Session(graph=valid_graph)

    if LOAD_MODEL:
        train_model.load(train_sess, 'log56')
    else:
        train_model.init(train_sess)
    
    start_time = time.time()
    it_start = train_model.get_global_step(train_sess)


    for it in range(it_start, ITERATION):
        train_loss, train_accu = train_model.train(train_sess)
        if (it+1)%DISPLAY_PERIOD==0:
            print('It:', it, 'Train loss:', train_loss, 'Train accu:', train_accu, 'Time:', time.time()-start_time)
            start_time = time.time()
        if (it+1)%VALID_PERIOD==0:
            train_model.save(train_sess, 'log56')
            valid_model.load(valid_sess, 'log56')
            valid_loss = []
            valid_accu = []
            for j in range(10):
                l, a = valid_model.valid(valid_sess)
                valid_loss.append(l)
                valid_accu.append(a)
            print('Validation loss:', sum(valid_loss)/len(valid_loss), 'Validation accu:', sum(valid_accu)/len(valid_accu))
            

if __name__=='__main__':
    main()