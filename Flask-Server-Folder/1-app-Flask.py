from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

import numpy as np
import cv2
# from PIL import Image

import base64
import uuid
import zmq

labels_file="/cxldata/projects/yolov4/coco.names"
LABELS = open(labels_file).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


# Define a flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/uploader', methods = ['POST'])
def upload_file():
    predictions=""

    if request.method == 'POST':        
        try:
            f = request.files['file']
            print("f type",type(f))

            # Save the uploaded file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'static','uploads', secure_filename(f.filename))
            print("file_path:",file_path)
            f.save(file_path)

            global img_str
            with open(file_path, "rb") as image_file:
                img_str = base64.b64encode(image_file.read())


            image = cv2.imread(file_path)
            print(image.shape)
            image= cv2.resize(image, (608, 608))
            print(type(image))

            context = zmq.Context()
            socket = context.socket(zmq.DEALER)
            _rid = "{}".format(str(uuid.uuid4()))
            socket.setsockopt(zmq.IDENTITY, _rid)
            socket.connect('tcp://localhost:5572')
            poll = zmq.Poller()
            poll.register(socket, zmq.POLLIN)
            obj = socket.send_json({"payload": img_str, "_rid": _rid})

            received_reply = False
            while not received_reply:
                sockets = dict(poll.poll(1000))
                if socket in sockets:
                    if sockets[socket] == zmq.POLLIN:
                        result_dict = socket.recv_json()                    
                        predictions = result_dict['preds']
                        received_reply = True            

                        for pred_i in predictions:
                            print(pred_i)
                            print("------------------")

                            x = pred_i['coordinates']['x']
                            y = pred_i['coordinates']['y']
                            h = pred_i['coordinates']['height']
                            w = pred_i['coordinates']['width']
                            confidence = pred_i['confidence']
                            class_label = pred_i['class'] 
                            color = pred_i['color']

                            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                            text = "{}: {:.4f}".format(class_label, confidence)
                            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #                     cv2.imwrite("prediction.png", image)
                                    # Save the uploaded file to ./uploads
                        detections_folder_path = os.path.join(basepath, 'static','detections', secure_filename(f.filename))
                        print("detections_folder_path:",detections_folder_path)
                        cv2.imwrite(detections_folder_path, image)
                    
#                     return render_template("upload.html", predictions=predictions, display_image=f.filename) 
                    return render_template("upload.html", display_image=f.filename) 
        except:
            return render_template("upload.html", msg="Please Upload an Image and then Submit") 
#             return render_template("upload.html", predictions="Please Upload an Image and then Submit", display_image=f.filename) 
#                     return render_template("upload.html", predictions=predictions, image_path=file_path) 

        socket.close()
        context.term()


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True,port="4112")
