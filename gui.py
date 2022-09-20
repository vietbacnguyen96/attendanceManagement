import os
import cv2
import torch
import threading
import base64
import requests
import time
import json                    
import unidecode
import argparse
from gtts import gTTS
import playsound
from datetime import datetime
import numpy as np
from caffe.ultra_face_opencvdnn_inference import inference, net as net_dnn, path

time.sleep(10)

parser = argparse.ArgumentParser(description='Face Recognition')
parser.add_argument('-db', '--debug', default='False',
        type=str, metavar='N', help='Turn on debug mode')
parser.add_argument('-rec', '--record', default='False',
        type=str, metavar='N', help='Record screen')
parser.add_argument('-fr', '--frequency', default='5',
        type=int, metavar='N', help='Frequency of face recognition')

args = parser.parse_args()
debug = False
record = False
frequency = args.frequency 
if args.debug == 'True':
	debug = True
if args.record == 'True':
	record = True

api = 'http://123.16.55.212:85/facerec'
window_name = 'Phần Mềm Điểm Danh - VKIST 2022'

record_time = datetime.fromtimestamp(time.time())
year = '20' + record_time.strftime('%y')
month = record_time.strftime('%m')
date = record_time.strftime('%d')
record_time = str(record_time).replace(' ', '_').replace(':', '_')


sound_dst_dir = path + 'sounds/'
if not os.path.exists(sound_dst_dir):
    os.makedirs(sound_dst_dir)
video_dst_dir = path + 'videos/'
if not os.path.exists(video_dst_dir):
    os.makedirs(video_dst_dir)
video_dst_dir += year + '/'
if not os.path.exists(video_dst_dir):
    os.makedirs(video_dst_dir)
video_dst_dir += month + '/'
if not os.path.exists(video_dst_dir):
    os.makedirs(video_dst_dir)
video_dst_dir += date + '/'
if not os.path.exists(video_dst_dir):
    os.makedirs(video_dst_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('\n**************************************************************************')
print('Running on ' + str(device))
print('**************************************************************************')
print('Debug: ' + str(debug))
print('Record: ' + str(record))
print('Frequency: ' + str(frequency))
print('Record sound folder: ' + str(sound_dst_dir))
print('Record video folder: ' + str(video_dst_dir))
print('**************************************************************************\n')

window_size = [1000, 730]

temp_boxes = []
predict_labels = []
queue = []
has_face = False

crop_image_size = 120
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.8
fontcolor = (0,255,0)

extend_pixel = 50

minimum_face_size = 60

box_size = 250
n_box = 1

openCvVidCapIds = []

for i in range(10):
    try:
        camI = cv2.VideoCapture(i)
        if camI is not None and camI.isOpened():
            openCvVidCapIds.append(i)
            camI.release()
        # camI.release()
    except:
        pass
print('\n*************************************')
print("All available camera's: " + str(openCvVidCapIds))
print('*************************************\n')
# define a video capture object
webcam = cv2.VideoCapture(openCvVidCapIds[0])
# webcam = cv2.VideoCapture(0)
print('\n*************************************')
if not webcam.isOpened():
    raise IOError("Webcam chưa được kết nối")
else:
    print("Đã kết nối với webcam id - " + str(openCvVidCapIds[0]))
print('*************************************\n')

print('webcam.get(cv2.CAP_PROP_FRAME_WIDTH): ' + str(webcam.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('webcam.get(cv2.CAP_PROP_FRAME_HEIGHT): ' + str(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('Window size: w = ', window_size[0], ' h = ', window_size[1])
frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
x_dis = int((frame_width - box_size * n_box) / (n_box + 1))
y_dis = int((frame_height - box_size) / 2)

cv2.namedWindow(window_name)
cv2.moveWindow(window_name, round((1280 - window_size[0]) * 0.5), 0)

temp_id = -2
temp_name = 'Khong co ai'
time_appear = time.time()
max_time_appear = 5
prev_frame_time = 0
new_frame_time = 0

cur_time = 0
max_times = 5

def remove_accent(text):
    return unidecode.unidecode(text)

def set_temp_value(new_id, new_name, is_reset):
	global temp_id, temp_name
	temp_id = new_id
	temp_name = new_name
	if is_reset and debug:
		print("--------------- Reset temp value")
	else:
		if debug:
			print("+++++++++++++++ Update temp value")

def check_first_time_appear(cur_id, cur_name, temp_id_):
	if cur_id != -1:
		if cur_id != temp_id_:
			if debug:
				print('cur_id: ' + str(cur_id) + ' temp_id: ' + str(temp_id_))
			set_temp_value(cur_id, cur_name, False)
			return True
		else:
			return False

def say_hello(content):
    unsign_content = remove_accent(content).replace(" ", "_")
    if not os.path.isfile(sound_dst_dir + unsign_content + ".mp3"):
        if debug:
            print("Creating " + unsign_content + ".mp3 file")
        tts = gTTS(content, tld = 'com.vn', lang='vi')


        tts.save(sound_dst_dir + unsign_content + ".mp3")
    # play(AudioSegment.from_mp3(path + unsign_content + ".mp3"))
    # subprocess.Popen(path + unsign_content + ".mp3")
    
    playsound.playsound(sound_dst_dir + unsign_content + ".mp3", True)

def face_recognize(frame):
    cur_hour = str(datetime.now()).split(" ")[1].split(":")[0]

    global temp_boxes, predict_labels, time_appear, max_time_appear, has_face, temp_id, temp_name, cur_time
    _, encimg = cv2.imencode(".jpg", frame)
    img_byte = encimg.tobytes()
    img_str = base64.b64encode(img_byte).decode('utf-8')
    new_img_str = "data:image/jpeg;base64," + img_str
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'charset': 'utf-8'}

    payload = json.dumps({"secret_key": "18e6e136-f722-4bbd-b39e-5975263035b7", "img": new_img_str})

    seconds = time.time()
    requests.post(api, data=payload, headers=headers, timeout=100)
    response = requests.post(api, data=payload, headers=headers, timeout=100)

    try:
        # print(response.json())
        for id, bb, name in zip(response.json()['result']['id'], response.json()['result']['bboxes'], response.json()['result']['identities']):
            response_time_s = time.time() - seconds
            print("Server's response time: " + "%.2f" % (response_time_s) + " (s)")
            bb = bb.split(' ')
            if check_first_time_appear(id, name, temp_id):
                time_appear = time.time()
                max_time_appear = 10

                non_accent_name = remove_accent(temp_name)
                if id > 0:
                    front_string = "Xin chào "
                    if int(cur_hour) > 15:
                        front_string = "Tạm biệt "
                    faceI = cv2.resize(frame[int(float(bb[1])): int(float(bb[3])), int(float(bb[0])): int(float(bb[2]))], (crop_image_size, crop_image_size))
                    predict_labels.append([non_accent_name, faceI])

                name_parts = temp_name.split(' - ')[0].split(' ')

                if non_accent_name.find(' Thi ') > -1 and len(name_parts) < 4:
                    print(front_string + name_parts[-1] + ' ' + name_parts[0] + '\n')  
                    say_hello(front_string + name_parts[-1] + ' ' + name_parts[0])
                else:
                    print(front_string + name_parts[-2] + ' ' + name_parts[-1] + '\n')  
                    say_hello(front_string + name_parts[-2] + ' ' + name_parts[-1])
            else:
                cur_time += 1
                if cur_time >= max_times:
                    temp_id = -2
                    temp_name = 'Khong co ai'
    except requests.exceptions.RequestException:
        print(response.text)

    return

size = (window_size[0], window_size[1])
# if record:
record_screen = cv2.VideoWriter(video_dst_dir + 'record_' + record_time + '.avi', 
                cv2.VideoWriter_fourcc(*'MJPG'),
                10, size)

count = 0
tm = cv2.TickMeter()
while True:
    tm.start()
    count += 1

    frame_show = np.ones((window_size[1],window_size[0],3),dtype='uint8') * 255    

    ret, orig_image = webcam.read()
    if orig_image is None:
        print("end")
        break

    distance_x = 20
    distance_y = 30
    final_frame = orig_image.copy()
    # temp_boxes, _, probs = inference(net_dnn, orig_image[y_dis: y_dis + box_size, x_dis: x_dis + box_size])
    temp_boxes, _, probs = inference(net_dnn, orig_image)

    for i, boxI in enumerate(temp_boxes):
        # boxI = boxI + [x_dis, y_dis, x_dis, y_dis]
        x1, y1, x2, y2 = int(boxI[0]), int(boxI[1]), int(boxI[2]), int(boxI[3])
        # if ((x2 - x1) * (y2 - y1)) / (box_size * box_size) > 0.2:
        final_frame = cv2.rectangle(final_frame,(x1, y1), (x2, y2),(0,255,0), 2)
    
    if len(temp_boxes) > 0:
        if (count % frequency) == 0:
            queue = [t for t in queue if t.is_alive()]
            if len(queue) < 3:
                queue.append(threading.Thread(target=face_recognize, args=(orig_image,)))
                queue[-1].start()
            count = 0

    # for i in range(0, n_box):
    #     final_frame = cv2.rectangle(final_frame,(int((x_dis + box_size) * i) + x_dis, y_dis), (int((x_dis + box_size) * i) + x_dis + box_size, y_dis + box_size),(255,0,0), 10)

    frame_show[1:frame_height + 1, 1 :frame_width + 1,:] = final_frame

    temp_labels = list(reversed(predict_labels))
    for i, labelI in enumerate(temp_labels):
        cv2.putText(frame_show, '{0}'.format(labelI[0]), (frame_width + distance_x, int((crop_image_size + distance_y) * i) + int(distance_y/1.5) ), fontface, fontscale, (100, 255, 0))
        if frame_width + distance_x + crop_image_size < window_size[0] and int((crop_image_size + distance_y) * i) + distance_y + crop_image_size < window_size[1]:
            frame_show[int((crop_image_size + distance_y) * i) + distance_y: int((crop_image_size + distance_y) * i) + distance_y + crop_image_size, frame_width + distance_x: frame_width + distance_x + crop_image_size, :] = labelI[1]

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))

    cv2.putText(frame_show, '{0} fps'.format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame_show, str(datetime.fromtimestamp(time.time())).split('.')[0], (20, frame_height - 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow(window_name, frame_show)
    # if record and len(temp_boxes) > 0:
    if len(temp_boxes) > 0:
        record_screen.write(frame_show)

    if len(predict_labels) > 3:
        predict_labels = predict_labels[1:]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()