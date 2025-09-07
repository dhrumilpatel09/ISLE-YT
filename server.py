import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
import base64
import math

app = Flask(__name__)
CORS(app)  # allow cross-origin requests

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

labels = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O",
          "P","Q","R","S","T","U","V","W","X","Y"]

offset = 20
imgSize = 300

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["image"]

        # Decode base64 image
        img_data = base64.b64decode(data.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        hands, img = detector.findHands(img)
        if not hands:
            return jsonify({"prediction": "None", "confidence": 0})

        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Safe crop
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            return jsonify({"prediction": "None", "confidence": 0})

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        return jsonify({
            "prediction": labels[index],
            "confidence": float(prediction[index])
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"prediction": "None", "confidence": 0})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
