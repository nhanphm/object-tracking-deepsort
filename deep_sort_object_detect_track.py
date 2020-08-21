import time
from typing import List
from absl import app
import cv2
import numpy as np
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# Definition of the parameters
max_cosine_distance = 0.6
nn_budget = None
nms_max_overlap = 0.8
weights_path = 'models/yolov3-tiny.weights'
configuration_path = 'models/yolov3-tiny.cfg'
pro_min = 0.5 # Setting minimum probability to eliminate weak predictions
threshold = 0.3 # Setting threshold for non maximum suppression

#initialize deep sort
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

net = cv2.dnn.readNet(weights_path, configuration_path)
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

with open("models/coco.names", "r") as f:
	class_names: List[str] = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(class_names), 3))  # This will be used later to assign colors for the bounding box for the detected objects

def get_objects_predictions(img):
	height, width = img.shape[:2]
	blob = cv2.dnn.blobFromImage(img, scalefactor = 1/255, size = (416, 416), mean= (0, 0, 0), swapRB = True, crop=False)
	net.setInput(blob)
	predictions = net.forward(output_layers)
	#print(predictions)
	return predictions,height, width

def get_box_dimentions(predictions,height, width, confThreshold = 0.5):
	class_ids = []
	confidences = []
	boxes = []
	for out in predictions:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)#Identifing the class type of the detected object by checking maximum confidence
			confidence = scores[class_id]
			if confidence > confThreshold:
				# Object detected
				center_x = int(detection[0] * width) #converting center_x with respect to original image size
				center_y = int(detection[1] * height)#converting center_y with respect to original image size
				w = int(detection[2] * width)#converting width with respect to original image size
				h = int(detection[3] * height)#converting height with respect to original image size
				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)
	return boxes,confidences,class_ids

def non_max_suppression(boxes,confidences,confThreshold = 0.5, nmsThreshold = 0.4):
	return cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

def detect_objects(img_path):
	predictions,height, width = get_objects_predictions(img_path)
	boxes,confidences,class_ids = get_box_dimentions(predictions,height, width)
	nms_indexes = non_max_suppression(boxes,confidences)
	boxes_x = []
	for i in range(len(boxes)):
		if i in nms_indexes:
			boxes_x.append(boxes[i])
	#img = draw_bouding_boxes(img_path,boxes,confidences,class_ids,nms_indexes,colors)
	return boxes_x

def check_laser_box(center_X,center_Y,box):
	(x1, y1, x2, y2) = [int(v) for v in box]
	if center_X > x1 and center_X < x2:
		if center_Y > y1 and center_Y < y2:
			return True
	return False

# Define cac tham so
video_path = 'test2.mp4'
max_distance = 100
input_h = 720
input_w = 1280
laser_box = [int(input_w / 2.0 - 200), int(input_h / 2.0 - 200), int(input_w / 2.0 + 200), int(input_h / 2.0 + 200)]
laser_line_color = (0, 255, 255)
def main(_argv):

	vid = cv2.VideoCapture(video_path)
	count = 0
	max_confidence = [0]*1000
	best_frame = []
	frame_index = 0
	while True:
		# Doc anh tu video
		_, frame = vid.read()
		if frame is None:
			time.sleep(0.1)
			count+=1
			if count < 3:
				continue
			else: 
				break
		if frame_index > 500 and frame_index < 1000:
			# Detect doi tuong
			predictions,height, width = get_objects_predictions(frame)
			boxes,confidences,class_ids = get_box_dimentions(predictions,height, width)

			#class_ids = class_ids[0]
			names = []
			for i in class_ids:
				names.append(class_names[i])
			names = np.array(names)
			features = encoder(frame, boxes)
			detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, confidences, names, features)]

			# run non-maxima suppresion
			boxs = np.array([d.tlwh for d in detections])
			scores = np.array([d.confidence for d in detections])
			classes = np.array([d.class_name for d in detections])
			indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
			detections = [detections[i] for i in indices]

			# Call the tracker
			tracker.predict()
			tracker.update(detections)

			for track in tracker.tracks:
				if not track.is_confirmed() or track.time_since_update > 1:
					continue
				bbox = track.to_tlbr()
				class_name = track.get_class()
				color = colors[int(track.track_id) % len(colors)]
				if track.confidence > max_confidence[track.track_id]:
					max_confidence[track.track_id] = track.confidence
					best_frame[int(track.track_id)] = frame
				cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
				cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
				cv2.putText(frame, class_name + "-" + str(track.track_id) + "-" + str(track.confidence),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

		# Frame
		cv2.rectangle(frame, (laser_box[0], laser_box[1]), (laser_box[2], laser_box[3]), laser_line_color, 2)
		cv2.putText(frame, "Frame: {:.2f}".format(frame_index), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
		cv2.imshow("Image", frame)
		frame_index +=1
		# Press Q on keyboard to  exit
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	vid.release()
	for i in range(len(best_frame)):
		cv2.imwrite("out/frame%d.jpg" % i, best_frame[i])
	
	cv2.destroyAllWindows

if __name__ == '__main__':
	try:
		app.run(main)
	except SystemExit:
		pass