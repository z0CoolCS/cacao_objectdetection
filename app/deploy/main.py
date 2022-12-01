from flask import Flask, request, jsonify
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2
import os

app = Flask(__name__)

#DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu')
CLASSES = [
    'background', 'Fitoftora', 'Monilia', 'Sana'
]

#model = tf.keras.models.load_model('saved_model_v2/my_model')
MODEL_DICT_FILE = 'model70.pth'


def create_model(num_classes):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

model = create_model(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(
     MODEL_DICT_FILE, map_location=DEVICE
))
model.eval()

def build_image(bytes_inp):
	file_bytes = np.fromstring(bytes_inp, np.uint8)
	img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED).astype(np.float32)
	img = cv2.imdecode(np.frombuffer(bytes_inp, np.uint8), -1).astype(np.float32)
	dim = (256, 256)
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	resized= np.transpose(resized, (2, 0, 1)).astype(np.float32)
	resized /= 255.0
	resized = torch.tensor(resized, dtype=torch.float).to(DEVICE)
	resized = torch.unsqueeze(resized, 0)
	return resized

def predict(inp):
	detection_threshold = 0.7
	with torch.no_grad():
		outputs = model(inp)
	outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

	boxes = outputs[0]['boxes'].data.numpy()
	scores = outputs[0]['scores'].data.numpy()
	boxes = boxes[scores >= detection_threshold].astype(np.int32)
	boxes = [list(box) for box in boxes]
	pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()[scores >= detection_threshold]]

	return boxes, pred_classes

@app.route("/", methods = ["GET", "POST"])
def index():
	if request.method == "POST":
		file = request.files.get("file")
		
		if file is None or file.filename == "":
			return jsonify({ "error" : "no file"})

		try:
			file = file.read()
			img = build_image(file)
			boxes, pred_classes = predict(img)
			print(boxes)
			print(pred_classes)
			return jsonify({ "boxes" : str(boxes), "pred_classes" : str(pred_classes)})
		except Exception as e:
			return jsonify({ "error" : e})

	return jsonify({ "ok" : "ok"})


if __name__ == "__main__":
	app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
