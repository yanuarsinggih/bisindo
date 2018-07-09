import thread
import itertools
import ctypes
import json

import pykinect
from pykinect import nui
from pykinect.nui import JointId
from pykinect.nui import SkeletonTrackingState

import pygame
from pygame.color import THECOLORS
from pygame.locals import *

from math import degrees,atan

import sys
from socket import *

serverHost = 'localhost'
serverPort = 9876

wordText = 'CONTOH KATA'

if len(sys.argv) > 1:
	serverHost = sys.argv[1]

#Create socket
#sSock = socket(AF_INET, SOCK_STREAM)

#Connect to server
#sSock.connect((serverHost, serverPort))
connect = False

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
				sSock.send(json.dumps(position_to_angle(data)))
			
def show_subs(subs):
	myfont = pygame.font.SysFont('Calibri', 50)
	text = myfont.render(subs, 1, THECOLORS["yellow"])
	text_w = text.get_rect().width
	text_h = text.get_rect().height
	screen.blit(text, ((VIDEO_WINSIZE[0]- text_w) / 2, VIDEO_WINSIZE[1] - text_h - 25))


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
	if not video_display:
		return

	with screen_lock:
		address = surface_to_array(screen)
		frame.image.copy_bits(address)
		del address
		if skeletons is not None and draw_skeleton:
			draw_skeletons(skeletons)
		show_subs(wordText)
		pygame.display.update()

if __name__ == '__main__':
	full_screen = False
	draw_skeleton = True
	video_display = True

	screen_lock = thread.allocate()
	
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
	print('	 d - Switch to depth view')
	print('	 v - Switch to video view')
	print('	 s - Toggle displaying of the skeleton')
	print('	 u - Increase elevation angle')
	print('	 j - Decrease elevation angle')
	print('esc - Quit program')

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
					sSock = socket(AF_INET, SOCK_STREAM)
					sSock.connect((serverHost, serverPort))
					connect = True
				else:
					sSock.shutdown(0)
					sSock.close()					
					#sSock = socket(AF_INET, SOCK_STREAM)
					connect = False