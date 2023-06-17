
# importing the library and packages
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse and parse the arguments
#The add_argument() method is used to specify the arguments that the script accepts. In this case, three arguments are defined:

#-i or --image: Specifies the path to the input image. It is marked as required, so the user must provide this argument.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")

#-c or --confidence: Specifies the minimum probability threshold to filter weak detections. It is optional and has a default value of 0.5.
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections, IoU threshold")

#-t or --threshold: Specifies the threshold for non-maxima suppression. It is optional and has a default value of 0.3.
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# reads the class labels from a file called "coco.names" and stores them in the LABELS list.
labelsPath = 'yolo-coco\\coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# generates random RGB color values for each class label and stores them in the COLORS array.
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

#Specifying the paths to the YOLO weights and model configuration
weightsPath = 'yolo-coco\\yolov3.weights'
configPath = 'yolo-coco\\yolov3.cfg'

#initializes the neural network with the pre-trained YOLO model.
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# function is used to read the image file and load it into memory
image = cv2.imread(args["image"])

## Get the height and width of the frame
(H, W) = image.shape[:2]

#returns a list containing the names of all layers in the network,
#  typically in the order they appear in the model.
ln = net.getLayerNames()

#returns a list of integers representing the indices of the unconnected output layers.
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Preprocess the image using cv2.dnn.blobFromImage
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)

#sets the blob as input to the neural network using 
net.setInput(blob)

# performs a forward pass through the network to obtain the output predictions
layerOutputs = net.forward(ln)

# Initialize three lists of detected bounding Boxes, CONFIDENCES, and
# class IDs, respectively
Boxes = []
CONFIDENCES = []
CLASSIDS = []

# iterates over each element in the layerOutputs list,
#  which contains the output blobs from the YOLO network.
for output in layerOutputs:
	#Within each layer output, the code iterates over each detection.
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		#scores from index 5 onwards in the detection array are extracted 
		scores = detection[5:]
		
		#function is used to find the index of the highest score,
		#  which represents the predicted class.
		#The class ID and confidence value corresponding to the highest score are assigned to the variables classID and confidence, respectively
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by 
		# checking if the confidence value is greater than the minimum confidence threshold specified by
		if confidence > args["confidence"]:
			#The bounding box coordinates (centerX, centerY, width, height) are extracted from the detection array using detection[0:4].
            #These coordinates are multiplied by an array [W, H, W, H] to scale them back relative to the size of the original input image.
			box = detection[0:4] * np.array([W, H, W, H])
			#resulting coordinates are then converted to integers
			(centerX, centerY, width, height) = box.astype("int")

		    #Using the scaled bounding box coordinates,
			#  calculates the top-left corner coordinates (x and y) of the bounding box.
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			#The top-left corner coordinates, width, height, confidence, 
			# and class ID of the current detection are appended to the Boxes, CONFIDENCES, and CLASSIDS lists, respectively.
			Boxes.append([x, y, int(width), int(height)])
			CONFIDENCES.append(float(confidence))
			CLASSIDS.append(classID)

# Non-Maxima Suppression (NMS) is applied to suppress weak, overlapping bounding Boxes. It helps eliminate redundant detections of the same object 
# and keeps only the most confident and non-overlapping Boxes.
idxs = cv2.dnn.NMSBoxes(Boxes, CONFIDENCES, args["confidence"],
	args["threshold"])

# checks if the length of idxs is greater than zero, ensuring that there is at least one valid detection after NMS
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (Boxes[i][0], Boxes[i][1])
		(w, h) = (Boxes[i][2], Boxes[i][3])

		# draw a bounding box rectangle and label on the image
		#A color is selected for the bounding box based on the class ID
		color = [int(c) for c in COLORS[CLASSIDS[i]]]

		#function is used to draw a rectangle around the object on the image 
		# using the bounding box coordinates.
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

		#The class label and confidence value are formatted into a text string using
		text = "{}: {:.4f}".format(LABELS[CLASSIDS[i]], CONFIDENCES[i])

		# function is used to overlay the text on the image at the top-left corner of the bounding box.
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

#It display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)