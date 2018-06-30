from selenium import webdriver

import tensorflow as tf
import numpy as np
from PIL import Image
import time
import random

import model5
import model6
import model56

test_graph5 = tf.Graph()
with test_graph5.as_default():
    model5 = model5.TestModel(1)
test_sess5 = tf.Session(graph=test_graph5, config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)))
model5.load(test_sess5, 'log5')

test_graph6 = tf.Graph()
with test_graph6.as_default():
    model6 = model6.TestModel(1)
test_sess6 = tf.Session(graph=test_graph6, config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)))
model6.load(test_sess6, 'log6')

test_graph56 = tf.Graph()
with test_graph56.as_default():
    model56 = model56.TestModel(1)
test_sess56 = tf.Session(graph=test_graph56, config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)))
model56.load(test_sess56, 'log56')


IDNumber = input('Please input your ID number: ') # 填入你的身分證字號
driver = webdriver.Chrome("./data/chromedriver") # chromedriver 路徑
correct, wrong = 0, 0

def int_to_cap(i):
    if 0<=i<10:
        return str(int(i))
    elif 10<=i<=35:
        return chr(i+55)
    else:
        print(i)
        assert(False)

def load_image(infilename) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="float32" ) / 255.0
    return data

for _ in range(100):
    driver.get('http://railway1.hinet.net/Foreign/TW/ecsearch.html')
    id_textbox = driver.find_element_by_id('person_id')
    id_textbox.send_keys(IDNumber)
    button = driver.find_element_by_css_selector('body > div.container > div.row.contents > div > form > div > div.col-xs-12 > button')
    button.click()
    driver.save_screenshot('tmp.png')
    location = driver.find_element_by_id('idRandomPic').location
    x, y = location['x'] + 5, location['y'] + 5
    img = Image.open('tmp.png')
    captcha = img.crop((x, y, x+200, y+60))
    captcha.convert("RGB").save('captcha.jpg', 'JPEG')
    # check is 5 or 6 digits
    cap_type = model56.predict(test_sess56, load_image('captcha.jpg'))
    print(cap_type)
    if cap_type[0]==1:
        predict = model5.predict(test_sess5, load_image('captcha.jpg'))
    else:
        predict = model6.predict(test_sess6, load_image('captcha.jpg'))
    ans = ''
    for i in range(len(predict)):
        ans += int_to_cap(predict[i][0])
    print(ans)
    captcha_textbox = driver.find_element_by_id('randInput')
    captcha_textbox.send_keys(ans)
    driver.find_element_by_id('sbutton').click()
    if "亂數號碼錯誤" in driver.page_source:
        wrong += 1
    else:
        correct += 1
    print("{:.4f}% (Correct{:d}-Wrong{:d})".format(correct/(correct+wrong)*100, correct, wrong))
    time.sleep(5)
