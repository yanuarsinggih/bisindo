import thread
import itertools
import ctypes
import json
from datetime import datetime
from scipy.stats import mode
import predict_mixed
import warnings

import pykinect
from pykinect import nui
from pykinect.nui import JointId
from pykinect.nui import SkeletonTrackingState

import pygame
from pygame.color import THECOLORS
from pygame.locals import *

from math import degrees,atan

import sys
import os
import cv2
import time
from socket import *

import threading
import numpy
from numpy import *

from net_config import ip_addr
 
class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        self._target = target
        self._args = args
        threading.Thread.__init__(self)
 
    def run(self):
        self._target(*self._args)

warnings.filterwarnings("ignore")

serverHost = 'localhost'
serverPort = 9876
gender = ''

if len(sys.argv) > 1:
    serverHost = sys.argv[1]

connect = False
save_image = False;

KINECTEVENT = pygame.USEREVENT
DEPTH_WINSIZE = 320,240
VIDEO_WINSIZE = 640,480
pygame.init()

SKELETON_COLORS = [THECOLORS["red"], 
                    THECOLORS["blue"], 
                    THECOLORS["green"], 
                    THECOLORS["orange"], 
                    THECOLORS["purple"], 
                    THECOLORS["yellow"], 
                    THECOLORS["violet"]]

wordText = 'http://192.168.10.33:5000/Bisindo/user120'
wordColor = THECOLORS["yellow"]

LEFT_ARM = (JointId.ShoulderCenter, 
            JointId.ShoulderLeft, 
            JointId.ElbowLeft, 
            JointId.WristLeft, 
            JointId.HandLeft)
RIGHT_ARM = (JointId.ShoulderCenter, 
             JointId.ShoulderRight, 
             JointId.ElbowRight, 
             JointId.WristRight, 
             JointId.HandRight)
LEFT_LEG = (JointId.HipCenter, 
            JointId.HipLeft, 
            JointId.KneeLeft, 
            JointId.AnkleLeft, 
            JointId.FootLeft)
RIGHT_LEG = (JointId.HipCenter, 
             JointId.HipRight, 
             JointId.KneeRight, 
             JointId.AnkleRight, 
             JointId.FootRight)
SPINE = (JointId.HipCenter, 
         JointId.Spine, 
         JointId.ShoulderCenter, 
         JointId.Head)
         
SKEL_DATA = []

skeleton_to_depth_image = nui.SkeletonEngine.skeleton_to_depth_image

def draw_skeleton_data(pSkelton, index, positions, width = 4):
    start = pSkelton.SkeletonPositions[positions[0]]
        
    for position in itertools.islice(positions, 1, None):
        next = pSkelton.SkeletonPositions[position.value]
        
        curstart = skeleton_to_depth_image(start, dispInfo.current_w, dispInfo.current_h) 
        curend = skeleton_to_depth_image(next, dispInfo.current_w, dispInfo.current_h)

        pygame.draw.line(screen, THECOLORS["orange"], curstart, curend, width)
        
        start = next

# recipe to get address of surface: http://archives.seul.org/pygame/users/Apr-2008/msg00218.html
if hasattr(ctypes.pythonapi, 'Py_InitModule4'):
    Py_ssize_t = ctypes.c_int
elif hasattr(ctypes.pythonapi, 'Py_InitModule4_64'):
    Py_ssize_t = ctypes.c_int64
else:
    raise TypeError("Cannot determine type of Py_ssize_t")

_PyObject_AsWriteBuffer = ctypes.pythonapi.PyObject_AsWriteBuffer
_PyObject_AsWriteBuffer.restype = ctypes.c_int
_PyObject_AsWriteBuffer.argtypes = [ctypes.py_object,
                                  ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.POINTER(Py_ssize_t)]

def surface_to_array(surface):
    buffer_interface = surface.get_buffer()
    address = ctypes.c_void_p()
    size = Py_ssize_t()
    _PyObject_AsWriteBuffer(buffer_interface,
                          ctypes.byref(address), ctypes.byref(size))
    bytes = (ctypes.c_byte * size.value).from_address(address.value)
    bytes.object = buffer_interface
    return bytes
    
def collect_data(data):
    global SKEL_DATA
    SKEL_DATA.append(json.loads(data))
    
def process_data():
    global SKEL_DATA, wordText
    pData = SKEL_DATA
    trueText = raw_input("Masukkan kata yang anda maksud: ")
    fileID = "{}-{}-{:%Y%M%d%H%m%S}".format(gender, trueText, datetime.now())
    with open("data/data-{}.json".format(fileID), 'w') as outfile:  
        json.dump(pData, outfile)
    
    tempWordText = 'http://192.168.10.33:5000/Bisindo/user120'
    wordText = 'Processing...'
    coll = []
    coll_m = []
    coll_f = []
    
    os.rename("static/Bisindo/user120/web_data.json", "static/Bisindo/user120/web_data_processed.json")
    arr_test = []
    for X_test in pData:
        dt = predict_mixed.predict_glvq([[X_test['RIGHT_HAND'][0], X_test['RIGHT_HAND'][1], X_test['LEFT_HAND'][0], X_test['LEFT_HAND'][1], 
        X_test['RIGHT_WRIST'][0], X_test['RIGHT_WRIST'][1], X_test['LEFT_WRIST'][0], X_test['LEFT_WRIST'][1],
        X_test['RIGHT_ELBOW'][0], X_test['RIGHT_ELBOW'][1], X_test['LEFT_ELBOW'][0], X_test['LEFT_ELBOW'][1],
        X_test['RIGHT_SHOULDER'][0], X_test['RIGHT_SHOULDER'][1], X_test['LEFT_SHOULDER'][0], X_test['LEFT_SHOULDER'][1]]])
        arr_test.append([X_test['RIGHT_HAND'][0], X_test['RIGHT_HAND'][1], X_test['LEFT_HAND'][0], X_test['LEFT_HAND'][1], 
        X_test['RIGHT_WRIST'][0], X_test['RIGHT_WRIST'][1], X_test['LEFT_WRIST'][0], X_test['LEFT_WRIST'][1],
        X_test['RIGHT_ELBOW'][0], X_test['RIGHT_ELBOW'][1], X_test['LEFT_ELBOW'][0], X_test['LEFT_ELBOW'][1],
        X_test['RIGHT_SHOULDER'][0], X_test['RIGHT_SHOULDER'][1], X_test['LEFT_SHOULDER'][0], X_test['LEFT_SHOULDER'][1]])
        coll.append(dt[0])
        coll_m.append(dt[1])
        coll_f.append(dt[2])
        #print dt
    
    # Predict DTW
    dtw_result = predict_mixed.predict_dtw(arr_test)
    
    # Predict HMM
    hmm_result = predict_mixed.predict_hmm(arr_test)
        
    # print coll
    #print ('Predict result mixed : ', mode(coll))
    #print ('Predict result male  : ', mode(coll_m))
    #print ('Predict result female: ', mode(coll_f))
    #dir(mode(coll))
    words = []
    if (len(coll) > 0):
        #words.append('{:>5}{:>10}{:>10}{:>10}'.format(
        #    'GLVQ', mode(coll).mode[0][0], mode(coll_m).mode[0][0], mode(coll_f).mode[0][0]
        #))
        #words.append('{:>5}{:>10}{:>10}{:>10}\n'.format('Model', 'Mixed', 'Male', 'Female'))
        #wordText = '\n'.join(words)
    
        # text = raw_input("Masukkan kata yang anda maksud: ")
    
        web_data = {
            'glvq' : {
                'mixed' : mode(coll).mode[0][0],
                'male' : mode(coll_m).mode[0][0],
                'female' : mode(coll_f).mode[0][0],
            },
            'dtw' : {
                'mixed' : dtw_result[0],
                'male' : dtw_result[1],
                'female' : dtw_result[2],
            },
            'hmm' : {
                'mixed' : hmm_result[0],
                'male' : hmm_result[1],
                'female' : hmm_result[2],
            },
        }
                
        with open("static/Bisindo/user120/web_data.json", 'w') as outfile:  
            json.dump(web_data, outfile)
            
        os.remove("static/Bisindo/user120/web_data_processed.json")
    
    #with open("result/result-{}.txt".format(fileID), 'w') as resfile:  
    #    resfile.write('Predict result mixed : {} with frequency {}/{}\n'.format(mode(coll).mode[0][0], mode(coll).count[0][0], len(coll)))
    #    resfile.write('Predict result male  : {} with frequency {}/{}\n'.format(mode(coll_m).mode[0][0], mode(coll_m).count[0][0], len(coll_m)))
    #    resfile.write('Predict result female: {} with frequency {}/{}\n'.format(mode(coll_f).mode[0][0], mode(coll_f).count[0][0], len(coll_f)))
    #    #print ('Saved to result/result-{}.txt'.format(fileID))
        
    SKEL_DATA = []
    wordText = tempWordText

def position_to_angle(data):
    angles = {
    #RIGHT_HAND
    'RIGHT_HAND':(
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.HandRight].z)/(data.SkeletonPositions[JointId.ShoulderCenter].x-data.SkeletonPositions[JointId.HandRight].x))),
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].y-data.SkeletonPositions[JointId.HandRight].y)/(data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.HandRight].z)))
    ),
    #LEFT_HAND
    'LEFT_HAND':(
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.HandLeft].z)/(data.SkeletonPositions[JointId.ShoulderCenter].x-data.SkeletonPositions[JointId.HandLeft].x))),
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].y-data.SkeletonPositions[JointId.HandLeft].y)/(data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.HandLeft].z)))
    ),
    #RIGHT_WRIST
    'RIGHT_WRIST':(
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.WristRight].z)/(data.SkeletonPositions[JointId.ShoulderCenter].x-data.SkeletonPositions[JointId.WristRight].x))),
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].y-data.SkeletonPositions[JointId.WristRight].y)/(data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.WristRight].z)))
    ),
    #LEFT_WRIST
    'LEFT_WRIST':(
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.WristLeft].z)/(data.SkeletonPositions[JointId.ShoulderCenter].x-data.SkeletonPositions[JointId.WristLeft].x))),
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].y-data.SkeletonPositions[JointId.WristLeft].y)/(data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.WristLeft].z)))
    ),
    #RIGHT_ELBOW
    'RIGHT_ELBOW':(
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.ElbowRight].z)/(data.SkeletonPositions[JointId.ShoulderCenter].x-data.SkeletonPositions[JointId.ElbowRight].x))),
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].y-data.SkeletonPositions[JointId.ElbowRight].y)/(data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.ElbowRight].z)))
    ),
    #LEFT_ELBOW
    'LEFT_ELBOW':(
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.ElbowLeft].z)/(data.SkeletonPositions[JointId.ShoulderCenter].x-data.SkeletonPositions[JointId.ElbowLeft].x))),
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].y-data.SkeletonPositions[JointId.ElbowLeft].y)/(data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.ElbowLeft].z)))
    ),
    #RIGHT_SHOULDER
    'RIGHT_SHOULDER':(
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.ShoulderRight].z)/(data.SkeletonPositions[JointId.ShoulderCenter].x-data.SkeletonPositions[JointId.ShoulderRight].x))),
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].y-data.SkeletonPositions[JointId.ShoulderRight].y)/(data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.ShoulderRight].z)))
    ),
    #LEFT_SHOULDER
    'LEFT_SHOULDER':(
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.ShoulderLeft].z)/(data.SkeletonPositions[JointId.ShoulderCenter].x-data.SkeletonPositions[JointId.ShoulderLeft].x))),
    degrees(atan((data.SkeletonPositions[JointId.ShoulderCenter].y-data.SkeletonPositions[JointId.ShoulderLeft].y)/(data.SkeletonPositions[JointId.ShoulderCenter].z-data.SkeletonPositions[JointId.ShoulderLeft].z)))
    )
    }
    return angles
    

def draw_skeletons(skeletons):
    for index, data in enumerate(skeletons):
        # draw the Head
        HeadPos = skeleton_to_depth_image(data.SkeletonPositions[JointId.Head], dispInfo.current_w, dispInfo.current_h) 
        draw_skeleton_data(data, index, SPINE, 10)
        pygame.draw.circle(screen, THECOLORS["orange"], (int(HeadPos[0]), int(HeadPos[1])), 20, 0)
    
        # drawing the limbs
        draw_skeleton_data(data, index, LEFT_ARM)
        draw_skeleton_data(data, index, RIGHT_ARM)
        # UNCOMMENT TO DRAW LOWER BODY
        #draw_skeleton_data(data, index, LEFT_LEG)
        #draw_skeleton_data(data, index, RIGHT_LEG)
        
        if data.tracking_state == SkeletonTrackingState.tracked:
            #print (position_to_angle(data))
            if connect:
                #sSock.send(json.dumps(position_to_angle(data)))
                collect_data(json.dumps(position_to_angle(data)))
            
def show_subs(subs):
    rSubs = subs.split('\n')
    for i in range(0, len(rSubs)):
        myfont = pygame.font.SysFont('Consolas', 20)
        text = myfont.render(rSubs[i], 1, THECOLORS["yellow"])
        text_w = text.get_rect().width
        text_h = text.get_rect().height
        # screen.blit(text, ((VIDEO_WINSIZE[0]- text_w) / 2, VIDEO_WINSIZE[1] - text_h - 25))
        screen.blit(text, (25, VIDEO_WINSIZE[1] - text_h - (25 * (i+1))))


def depth_frame_ready(frame):
    if video_display:
        return

    with screen_lock:
        address = surface_to_array(screen)
        frame.image.copy_bits(address)
        del address
        if skeletons is not None and draw_skeleton:
            draw_skeletons(skeletons)
        show_subs(wordText)
        pygame.display.update()    

def video_frame_ready(frame):
    global save_image
    if not video_display:
        return

    with screen_lock:
        fileID = "{:%Y%M%d%H%m%S%f}".format(datetime.now())
        address = surface_to_array(screen)
        frame.image.copy_bits(address)
        current_directory = os.getcwd()
        del address
        if skeletons is not None and draw_skeleton:
            draw_skeletons(skeletons)
        show_subs(wordText)
        pygame.display.update()
        
        # try:
        frame_width,frame_height = VIDEO_WINSIZE
        # save to Folder
        folder = 'RGB_images\\'
        path = current_directory+str('\\')+folder
        image_name = 'color_image_{}.jpg'.format(fileID)
        file = os.path.join(path,image_name)       
        # if frame_height == int(frame.image.height) and frame_width == int(frame.image.width):
        
        rgb = numpy.empty((frame.image.height, frame.image.width, 4), numpy.uint8) 
        frame.image.copy_bits(rgb.ctypes.data) #copy the bit of the image to the array
        cv2.imwrite(os.path.join(path,image_name), rgb)
        
        if len(os.listdir("RGB_images")) > 10:
            for root, dirs, files in os.walk("RGB_images", topdown=False):
                for name in files:
                    fileName = os.path.join(root, name)
                    try:
                        os.remove(fileName)
                    except:
                        pass
                    break
                break
        # except:
        #    pass
    
if __name__ == '__main__':
    full_screen = False
    draw_skeleton = True
    video_display = True

    screen_lock = thread.allocate()
    
    # Cleanse RGB_images folder
    try:
        for root, dirs, files in os.walk("RGB_images", topdown=False):
            for name in files:
                fileName = os.path.join(root, name)
                os.remove(fileName)
                break
            break
    except:
        pass
        
    wordText = 'http://{}:5000/Bisindo/user120/'.format(ip_addr)
    
    icon = pygame.image.load('icon-sample.png')
    pygame.display.set_icon(icon)

    screen = pygame.display.set_mode(VIDEO_WINSIZE,0,32)    
    pygame.display.set_caption('Python Kinect Streaming')
    skeletons = None
    screen.fill(THECOLORS["black"])    

    kinect = nui.Runtime()
    kinect.skeleton_engine.enabled = True
    def post_frame(frame):
        try:
            pygame.event.post(pygame.event.Event(KINECTEVENT, skeletons = frame.SkeletonData))
        except:
            # event queue full
            pass

    kinect.skeleton_frame_ready += post_frame
    
    kinect.depth_frame_ready += depth_frame_ready    
    kinect.video_frame_ready += video_frame_ready    
    
    kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution320x240, nui.ImageType.Depth)
    kinect.video_stream.open(nui.ImageStreamType.Video, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Color)
    
    print('Controls: ')
    print('     d - Switch to depth view')
    print('     v - Switch to video view')
    print('     s - Toggle displaying of the skeleton')
    print('     u - Increase elevation angle')
    print('     j - Decrease elevation angle')
    print('esc - Quit program')
    
    while gender.lower() != 'p' and gender.lower() != 'w': 
        gender = raw_input("Masukkan jenis kelamin anda (P/W): ")

    # main game loop
    done = False    

    while not done:
        e = pygame.event.wait()
        dispInfo = pygame.display.Info()
        if e.type == pygame.QUIT:
            done = True
            break
        elif e.type == KINECTEVENT:
            skeletons = e.skeletons
            if draw_skeleton:
                draw_skeletons(skeletons)
                pygame.display.update()
        elif e.type == KEYDOWN:
            if e.key == K_ESCAPE:
                done = True
                break
            elif e.key == K_d:
                with screen_lock:
                    screen = pygame.display.set_mode(DEPTH_WINSIZE,0,16)
                    video_display = False
            elif e.key == K_v:
                with screen_lock:
                    screen = pygame.display.set_mode(VIDEO_WINSIZE,0,32)    
                    video_display = True
            elif e.key == K_s:
                draw_skeleton = not draw_skeleton
            elif e.key == K_u:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle + 2
            elif e.key == K_j:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle - 2
            elif e.key == K_x:
                kinect.camera.elevation_angle = 2
            elif e.key == K_SPACE:
                if not connect:
                    #sSock = socket(AF_INET, SOCK_STREAM)
                    #sSock.connect((serverHost, serverPort))
                    connect = True
                    wordColor = THECOLORS["green"]
                else:
                    #sSock.shutdown(0)
                    #sSock.close()                    
                    connect = False                    
                    wordColor = THECOLORS["yellow"]
                    t1 = FuncThread(process_data)
                    t1.start()
                    t1.join()