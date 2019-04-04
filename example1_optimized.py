import cv2 as cv 
import threading 
import time
import argparse
import queue



parser = argparse.ArgumentParser(
        description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--inp', help='Path to image or video. Integer for camera/USB number. Skip to capture frames from camera', default =0)
parser.add_argument('--task', help='task to be performed',default='edge_detect', required=False)
parser.add_argument('--config', help='Name of config file or Path to .prototxt',default='deploy.prototxt' ,required=False)
parser.add_argument('--model', help='Name of model or Path to .model', default ='hed_pretrained_bsds.caffemodel',required=False)
parser.add_argument('--framework', help='Name of framework', default = 'caffe',required=False)

args = parser.parse_args()




class video_inferencing():
	def __init__(self,task,config,model,framework,frame):
			self.task=task
			self.config=config
			self.model=model
			self.framework=framework
			self.frame=frame
			self.output_queue=queue.Queue()
	def start(self):
		
		threading.Thread(target=self.run_model,args=(self.task,self.config,self.model,self.framework,self.frame,self.input_queue,)).start()
		return self

	#def blob_parameters(self,task):
		# load parameters for cv.dnn.blobFromImage based on task defined from perviously created file
		# else get input for different parameters for the function
	
		
	def load_files(self,task,config,model):
		# if pathto .prototext file and .model defined, directly load that 
		# else  read and download the various .prototext, .model files based on text and framework defined from a previously defined file
		# for this sample path is already provided
		print("loading_files")
		return self.config,self.model
	def run_model(self,task,config,model,framework,frame):
		print("running_model")

		config,model=self.load_files(self.config,self.task,self.model)
		

		blob= cv.dnn.blobFromImage(self.frame, scalefactor=1.0, size=(500,500),mean=(104.00698793, 116.66876762, 122.67891434),swapRB=False, crop=False)

		print("running_model2")
		# fixed parameters for edge detetion
		net = cv.dnn.readNet(config,model,framework)
		
		net.setInput(blob)
		out = net.forward()
		output.queue.put(out)
		return out
	def display_image(self,output,frame, task):
		# specific to edge_detection right now
		#change to make it general to any task
		out = output[0, 0]
		out = cv.resize(out,(self.frame.shape[1], self.frame.shape[0]))
		cv.imshow(self.task, out)



class optimized_video_read():
	def __init__(self, inp):
		self.inp=inp
		self.video = cv.VideoCapture(self.inp)
		(self.isFrame,self.frame)=self.video.read()
		self.flag=False
		self.input_queue=queue.Queue(1)
		self.track_queue=queue.Queue(1)

	def start(self):
		#print("threading")
		threading.Thread(target=self.update,args=(self.input_queue,)).start()
		return self
	def update(self,input_queue):
		while True:
			#print("update")
			if self.flag:
				print("no frame")
				return
			(self.isFrame,self.frame)=self.video.read()
			self.input_queue.put(self.frame)
			self.track_queue.put(self.isFrame)
			#print("update end")
	def read(self):
		#print("read")
		
		return self.track_queue.get(),self.input_queue.get()

	def stop(self):
		#print("stop")
		self.flag=True
		self.video.release()

config=args.config
model=args.model
task=args.task
framework=args.framework
inp=args.inp


stream=optimized_video_read(inp).start()
while cv.waitKey(1)<0:
	start=time.time()

	isFrame, frame = stream.read()
	if not isFrame:
		stream.stop()
		cv.waitKey()
		break
	cv.imshow('Input', frame)
	edge_detect = video_inferencing(task,config,model,framework,frame)
	output= edge_detect.run_model(task, config,model,framework,frame)
	edge_detect.display_image(output,frame,"edge_detect")
	stop = time.time()
	print("time elapsed",(stop-start))


cv.destroyAllWindows()