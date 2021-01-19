from io import BytesIO
from PIL import Image
import threading
import zmq
from base64 import b64decode
import numpy as np

import glob
import random
# import darknet
import time
import cv2

print ("Load Start", time.asctime())
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4

labels_file="/cxldata/projects/yolov4/coco.names"
LABELS = open(labels_file).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")



weights="/cxldata/projects/yolov4/yolov4.weights"
config="/cxldata/projects/yolov4/yolov4.cfg"
net = cv2.dnn.readNetFromDarknet(config, weights)

# labelsFile="./cfg/obj.names"
# net = cv2.dnn.readNetFromDarknet("./cfg/yolov4-obj.cfg", "./mcq.weights", )

LABELS = open(labels_file).read().strip().split("\n")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


print ("Load Ended", time.asctime())

# img_width = 416
# img_height = 416



class Server(threading.Thread):
    def __init__(self):
        self._stop = threading.Event()
        threading.Thread.__init__(self)

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        context = zmq.Context()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind('tcp://*:5572')

        backend = context.socket(zmq.DEALER)
        backend.bind('inproc://backend_endpoint')

        poll = zmq.Poller()
        poll.register(frontend, zmq.POLLIN)
        poll.register(backend,  zmq.POLLIN)

        while not self.stopped():
            sockets = dict(poll.poll())
            if frontend in sockets:
                if sockets[frontend] == zmq.POLLIN:
                    _id = frontend.recv()
                    json_msg = frontend.recv_json()

                    handler = RequestHandler(context, _id, json_msg)
                    handler.start()

            if backend in sockets:
                if sockets[backend] == zmq.POLLIN:
                    _id = backend.recv()
                    msg = backend.recv()
                    frontend.send(_id, zmq.SNDMORE)
                    frontend.send(msg)

        frontend.close()
        backend.close()
        context.term()

class RequestHandler(threading.Thread):
    def __init__(self, context, id, msg):

        """
        RequestHandler
        :param context: ZeroMQ context
        :param id: Requires the identity frame to include in the reply so that it will be properly routed
        :param msg: Message payload for the worker to process
        """
        threading.Thread.__init__(self)
        print("--------------------Entered requesthandler--------------------")
        self.context = context
        self.msg = msg
        self._id = id


    def process(self, obj):

        print ("Start of Processing", time.asctime())
        imgstr = obj['payload']

        img = Image.open(BytesIO(b64decode(imgstr)))
        print("type of img:", type(img))

        if img.mode != "RGB":
            img = img.convert("RGB")
        img = np.array(img)
        print("type of img:", type(img))
        img= cv2.resize(img, (608, 608))
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (608, 608), swapRB=True, crop=False)
        net.setInput(blob)
        print ("Pre processing", time.asctime())
        layerOutputs = net.forward(ln)
        print ("Post After inference", time.asctime())
        return_dict ={}

        boxes = []
        confidences = []
        classIDs = []
        (H, W) = img.shape[:2]
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.3:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        print ("Confidence extracted", time.asctime())
        output_bboxes = []
        print ("classIDs", classIDs)
        for j in range(len(LABELS)):
            tmpboxes = []
            tmpconf =[]
            tmpClassIDs=[]

            print (j, LABELS[j])
            for i in range(len(boxes)):
                if classIDs[i] == j:
                    tmpClassIDs.append(j)
                    tmpboxes.append(boxes[i])
                    tmpconf.append(confidences[i])

            idxs = cv2.dnn.NMSBoxes(tmpboxes, tmpconf, 0.5, 0.3)

            print ("NMS steps done", time.asctime())
            print (len(tmpboxes), idxs)
            if len(idxs) > 0:
                for k in idxs.flatten():
                    (x, y) = (tmpboxes[k][0], tmpboxes[k][1])
                    (w, h) = (tmpboxes[k][2], tmpboxes[k][3])
                    
                    color = [int(c) for c in COLORS[j]]
                    
                    output_bboxes.append({
                    'class': LABELS[j],
                    'classId': str(j),
                    'color':color,
                    'confidence': tmpconf[k],
                    'coordinates': {
                        'x_center': x + w/2,
                        'y_center': y + h/2,
                        'width': w,
                        'height': h,
                        'x':x,
                        'y':y
                    }})

                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[j], tmpconf[k])
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            #print (detections)

        return_dict["preds"] = output_bboxes
        cv2.imwrite("prediction.png", img)
        return return_dict

    def run(self):
        # Worker will process the task and then send the reply back to the DEALER backend socket via inproc
        worker = self.context.socket(zmq.DEALER)
        worker.connect('inproc://backend_endpoint')
        #print('Request handler started to process %s\n' % self.msg)

        # Simulate a long-running operation
        output = self.process(self.msg)

        worker.send(self._id, zmq.SNDMORE)
        worker.send_json(output)
        del self.msg

        print('Request handler quitting.\n')
        worker.close()


def main():
    # Start the server that will handle incoming requests
    print ("Ready for Server Start", time.asctime())
    server = Server()
    server.start()
    print ("Server started", time.asctime())

if __name__ == '__main__':
    main()