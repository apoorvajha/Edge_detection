import cv2 as cv 
import time
import argparse
parser = argparse.ArgumentParser(
description='This sample shows how to define custom OpenCV deep learning layers in Python. '
'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--inp', help='Path to image or video. Skip to capture frames from camera',default=0)
parser.add_argument('--task', help='task to be performed',default='edge_detect', required=False)
#parser.add_argument('--config', help='Name of config file or Path to .prototxt',default ,required=False)
parser.add_argument('--model', help='Name of model or Path to model', default ='Holistically-Nested Edge Detection',required=False)
parser.add_argument('--framework', help='Name of framework', default = 'caffe',required=False)
args = parser.parse_args()
class fileLoader():
	def __init__(self,task,model,framework):
		self.task=task
		self.model=model
		self.framework=framework
		self.config=None
		self.modelFile=None
	def loadFiles(self,task,model,framework):
		# if path to .prototext file and .model defined, directly load that 
		# else  read and download the various .prototext, model files by mapping the task, framework with the URL requires
		#by default config neendnt be given but for this task we are uploading the config file
		print("loading_files")
		self.config="deploy.prototxt"
		if model=="Holistically-Nested Edge Detection":
			self.modelFile='hed_pretrained_bsds.caffemodel'

		return self.config,self.modelFile
#class hyperParameters():
	#def __init__(self,param):
		# load parameters for cv.dnn.blobFromImage based on task defined from perviously created file
		# else get input for different parameters for the function
class loadModel(fileLoader):
	def __init__(self,task,model,framework):
		fileLoader.__init__(self,task,model,framework)
		self.config=None
		self.modelFile=None
	def readNetModel(self,task,model,framework):
		self.config,self.modelFile=super().loadFiles(task,model,framework)
		net = cv.dnn.readNet(self.config,self.modelFile,self.framework)
		return net
class preProcessing():
	def __init__(self,frame):
		self.frame=frame

		#self.para=param inherit parameter from hyperParameters class
		#def framProccess():
	def blobConvert(self,frame):
		blob= cv.dnn.blobFromImage(self.frame, scalefactor=1.0, size=(500,500),
		mean=(104.00698793, 116.66876762, 122.67891434),
		swapRB=False, crop=False)
		return blob
class runModel(preProcessing,loadModel):
	def __init__(self,task,model,framework,frame):
		preProcessing.__init__(self,frame)
		loadModel.__init__(self,task,model,framework)
		self.net=None
		self.blob=None
		self.out=None
	def run(self,task,model,framework,frame):
		net = super().readNetModel(task,model,framework)
		blob= super().blobConvert(frame)
		net.setInput(blob)
		out = net.forward()
		return out
class videoOutput(runModel):
	def __init__(self,task,model,framework,frame):
		runModel.__init__(self,task,model,framework,frame)
		self.out=None
		# specific to edge_detection right now
		#change to make it general to any task
	def displayImage(self,task,model,framework,frame):
		print("displaying image")
		out=super().run(task,model,framework,frame)
		out = out[0, 0]
		out = cv.resize(out,(self.frame.shape[1], self.frame.shape[0]))
		cv.imshow(self.task, out)
	#def videoStorage():
model=args.model
task=args.task
framework=args.framework
inp=args.inp
cap = cv.VideoCapture(inp )
while cv.waitKey(1) < 0:
	start=time.time()
	print("reading frame")
	hasFrame, frame = cap.read()
	if not hasFrame:
		cv.waitKey()
		print("no frame")
		break
	cv.imshow('Input', frame)
	edge_detect = videoOutput(task,model,framework,frame)
	output= edge_detect.displayImage(task, model,framework,frame)
	stop = time.time()
	print("time elapsed",(stop-start))
cap.release()
cv.destroyAllWindows()




