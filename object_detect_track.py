import cv2
import math
import numpy as np

weights_path = 'models/yolov3-tiny.weights'
configuration_path = 'models/yolov3-tiny.cfg'

pro_min = 0.5 # Setting minimum probability to eliminate weak predictions

threshold = 0.3 # Setting threshold for non maximum suppression
net = cv2.dnn.readNet(weights_path, configuration_path)
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()
with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))  # This will be used later to assign colors for the bounding box for the detected objects

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


# Ham check xem old hay new
def is_old(center_Xd, center_Yd, boxes):
    for box_tracker in boxes:
        (xt, yt, wt, ht) = [int(c) for c in box_tracker]
        center_Xt, center_Yt = int((xt + (xt + wt)) / 2.0), int((yt + (yt + ht)) / 2.0)
        distance = math.sqrt((center_Xt - center_Xd) ** 2 + (center_Yt - center_Yd) ** 2)

        if distance < max_distance:
            return True
    return False


def get_box_info(box):
    (x, y, w, h) = [int(v) for v in box]
    center_X = int((x + (x + w)) / 2.0)
    center_Y = int((y + (y + h)) / 2.0)
    return x, y, w, h, center_X, center_Y

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
laser_line = input_w - 200
laser_box = [int(input_w / 2.0 - 200), int(input_h / 2.0 - 200), int(input_w / 2.0 + 200), int(input_h / 2.0 + 200)]

#net = cv2.dnn.readNetFromCaffe(prototype_url, model_url)
vid = cv2.VideoCapture(video_path)

# Khoi tao tham so
frame_count = 0
car_number = 0
obj_cnt = 0
curr_trackers = []

while vid.isOpened():

	laser_line_color = (0, 0, 255)
	boxes = []

    # Doc anh tu video
	ret, frame = vid.read()
	if ret == True:
        # Duyet qua cac doi tuong trong tracker
		old_trackers = curr_trackers
		curr_trackers = []

		for car in old_trackers:

			# Update tracker
			tracker = car['tracker']
			(_, box) = tracker.update(frame)
			boxes.append(box)

			new_obj = dict()
			new_obj['tracker_id'] = car['tracker_id']
			new_obj['tracker'] = tracker

			# Tinh toan tam doi tuong
			x, y, w, h, center_X, center_Y = get_box_info(box)
			
			# Ve hinh chu nhat quanh doi tuong
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			#Dat ten cho doi tuong
			cv2.putText(frame, "object"+str(new_obj['tracker_id']), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 3)

			# Ve hinh tron tai tam doi tuong
			cv2.circle(frame, (center_X, center_Y), 4, (0, 255, 0), -1)

			# So sanh tam doi tuong voi duong laser line
			if center_X > laser_line:
				# Neu vuot qua thi khong track nua ma dem xe
				laser_line_color = (0, 255, 255)
				car_number += 1
			else:
				# Con khong thi track tiep
				curr_trackers.append(new_obj)

		# Thuc hien object detection moi 5 frame
		if frame_count % 5 == 0:
			# Detect doi tuong
			#boxes_d = get_object(net, frame)
			boxes_d = detect_objects(frame)
			for box in boxes_d:
				old_obj = False

				xd, yd, wd, hd, center_Xd, center_Yd = get_box_info(box)

				if center_Xd <= laser_line - max_distance:

					# Duyet qua cac box, neu sai lech giua doi tuong detect voi doi tuong da track ko qua max_distance thi coi nhu 1 doi tuong
					if not is_old(center_Xd, center_Yd, boxes):
						cv2.rectangle(frame, (xd, yd), ((xd + wd), (yd + hd)), (0, 255, 0), 2)

						# Tao doi tuong tracker moi
						tracker = cv2.TrackerMOSSE_create()

						obj_cnt += 1
						new_obj = dict()
						tracker.init(frame, tuple(box))

						new_obj['tracker_id'] = obj_cnt
						new_obj['tracker'] = tracker
						#Dat ten cho doi tuong
						cv2.putText(frame, "object"+str(new_obj['tracker_id']), (xd, yd - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 3)
						
						curr_trackers.append(new_obj)
			# Duyet qua cac doi tuong da track, neu doi tuong nao khong co detect hien tai thi xoa khoi track


		# Tang frame
		frame_count += 1

		# Hien thi so xe

		cv2.putText(frame, "Object number: " + str(car_number), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255 , 0), 2)
		cv2.putText(frame, "Press Q to quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

		# Draw laser line
		#cv2.line(frame, (0, laser_line), (input_w, laser_line), laser_line_color, 2)
		#cv2.line(frame, (laser_line,0), (laser_line,input_h), laser_line_color, 2)
		cv2.rectangle(frame, (laser_box[0], laser_box[1]), (laser_box[2], laser_box[3]), laser_line_color, 2)
		cv2.putText(frame, "Laser line", (10, laser_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, laser_line_color, 2)

		# Frame
		cv2.imshow("Image", frame)
		# Press Q on keyboard to  exit 
		if cv2.waitKey(25) & 0xFF == ord('q'): 
			break
	else:
		break
vid.release()
cv2.destroyAllWindows
