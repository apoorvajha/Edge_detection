import cv2 as cv 

import time
import argparse



parser = argparse.ArgumentParser(
        description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--inp', help='Path to image or video. Skip to capture frames from camera',default=0)
parser.add_argument('--task', help='task to be performed',default='edge_detect', required=False)
parser.add_argument('--config', help='Name of config file or Path to .prototxt',default='deploy.prototxt' ,required=False)
parser.add_argument('--model', help='Name of model or Path to model', default ='hed_pretrained_bsds.caffemodel',required=False)
parser.add_argument('--framework', help='Name of framework', default = 'caffe',required=False)

args = parser.parse_args()




class video_inferencing():
	def __init__(self,task,config,model,framework,frame):
			self.task=task
			self.config=config
			self.model=model
			self.framework=framework
			self.frame=frame

	#def blob_parameters(self,task):
		# load parameters for cv.dnn.blobFromImage based on task defined from perviously created file
		# else get input for different parameters for the function
	
		
	def load_files(self,task,config,model):
		# if pathto .prototext file and .model defined, directly load that 
		# else  read and download the various .prototext, model files based on text and framework defined from a previously defined file
		# for this sample path is already provided
		print("loading_files")
		return self.config,self.model
	def run_model(self,task,config,model,framework,frame):
		print("running_model")

		config,model=self.load_files(self.config,self.task,self.model)


		blob= cv.dnn.blobFromImage(self.frame, scalefactor=1.0, size=(500,500),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)

         # fixed parameters for edge detetion
		net = cv.dnn.readNet(config,model,framework)
		net.setInput(blob)
		out = net.forward()
		
		return out
	def display_image(self,output,frame, task):
		# specific to edge_detection right now
		#change to make it general to any task
		print("displaying image")
		out = output[0, 0]
		out = cv.resize(out,(self.frame.shape[1], self.frame.shape[0]))
		cv.imshow(self.task, out)



	

config=args.config
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


	
	edge_detect = video_inferencing(task,config,model,framework,frame)
	
	output= edge_detect.run_model(task, config,model,framework,frame)
	
	

	edge_detect.display_image(output,frame,"edge_detect")

	stop = time.time()
	print("time elapsed",(stop-start))
cap.release()
cv.destroyAllWindows()




