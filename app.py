from fileinput import filename
import os
import uuid
import flask
import urllib
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , send_file
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread

global capture,rec_frame, grey, switch, neg, face, rec, out 
global path_
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# Edit dengan IP pada IP Webcam, contoh :
# camera = cv2.VideoCapture("https://192.168.0.102:8080/video ")
camera = cv2.VideoCapture("")

app = Flask(__name__)

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_obj = core.Model.load(os.path.join(BASE_DIR , 'model_weights.pth'),['Biji Kakao'])
model_clf = load_model(os.path.join(BASE_DIR , 'model_mobileNet_100_rlr_86.h5'))

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

reverse_mapping_col={0:(0,0,255),1:(0,165,225),2:(0,255,255),3:(0,255,0)}
#reverse_mapping_col={0:(0,255,0),1:(165,255,0),2:(255,255,0),3:(255,0,0)}
#reverse_mapping_col={0:'red',1:'orange',2:'yellow',3:'green'}
def mapper_col(value):
    return reverse_mapping_col[value]

reverse_mapping={0:'Insect',1:'Other',2:'Mould',3:'Normal'}
def mapper(value):
    return reverse_mapping[value]

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame, path_
    while True:
        success, frame = camera.read() 
        frame = cv2.flip(frame,1)
        
        frame = cv2.resize(frame, (500,500), interpolation= cv2.INTER_LINEAR)
        if success:  
            if(capture):
                capture=0
                frame = cv2.flip(frame,1)
                now = datetime.datetime.now()
                path_ = os.path.sep.join(['shots', "shot_{}.jpg".format(str(now).replace(":",''))])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(path_, frame)
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass
        

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take/success',methods=['POST','GET'])
def tasks():
    error=''
    global switch,camera, path_
    # switch=0
    # camera.release()
    # cv2.destroyAllWindows()
    if request.method == 'POST':
        if request.form.get('click') == 'Capture and Detect':
            global capture
            capture=1
            time.sleep(1)
            list_ = os.listdir('shots')
            list_ = sorted(list_, reverse=True)
            img_path = os.path.join( 'shots', str(list_[0]))
            img = 'result.png'
            class_result , percentage, time_, time__, time___ = predict(img_path , model_obj, model_clf)
            predictions = {
                    "class1":class_result[0],
                    "class2":class_result[1],
                    "class3":class_result[2],
                    "class4" : class_result [3],
                    "perc1": percentage[0],
                    "perc2": percentage[1],
                    "perc3": percentage[2],
                    "perc4": percentage[3],
                    "time" : time_,
                    "time1" : time__,
                    "time2" : time___
            }
            if(len(error) == 0):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('take.html' , error = error) 
    elif request.method=='GET':
        return render_template('take.html')
    return render_template('take.html')

@app.route('/take')
def take() :
    global switch,camera
    camera = cv2.VideoCapture(2)
    switch=1
    return render_template('take.html')


def predict(filename , model_obj, model_clf):
    t = datetime.datetime.now()
    thresh=0.8
    image_ = utils.read_image(filename)
    image_ = cv2.resize(image_, (1200,1200), interpolation= cv2.INTER_LINEAR)
    predictions = model_obj.predict(image_)
    labels, boxes, scores = predictions
    filtered_indices=np.where(scores>thresh)
    filtered_boxes=boxes[filtered_indices]
    area = filtered_boxes.numpy()
    result = []
    insect = 0
    other = 0
    mould = 0
    normal = 0
    percentage = [None] * 4
    t0 = datetime.datetime.now()
    for a in area :
        xmin = np.floor(a[0]).astype(int)
        ymin = np.floor(a[1]).astype(int)
        xmax = np.floor(a[2]).astype(int)
        ymax = np.floor(a[3]).astype(int)
        crop_img = image_[ymin:ymax, xmin:xmax]
        cv2.imwrite(os.path.join(BASE_DIR,"tmp/temp.png"),crop_img)
        img_=load_img(os.path.join(BASE_DIR,"tmp/temp.png"), target_size=(224,224))
        image=img_to_array(img_)
        prediction_image=np.array(crop_img)
        prediction_image= np.expand_dims(image, axis=0)
        prediction=model_clf.predict(prediction_image)
        score = round(np.max(prediction),2)
        value=np.argmax(prediction)
        result.append(value)
        label = mapper(value)+"("+str(score)+")"
        cv2.putText(image_, label, (xmin+13, ymin + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, mapper_col(value), 2)
        cv2.rectangle(image_, (xmin, ymin), (xmax, ymax), mapper_col(value), 3)
        if mapper(value) == "Insect" :
            insect = insect + 1
        elif mapper(value) == "Other" :
            other = other + 1
        elif mapper(value) == "Mould" :
            mould = mould + 1
        elif mapper(value) == "Normal" :
            normal = normal + 1

    #fig = plt.figure(figsize =(10, 10))
    #plt.imshow(image_)
    #fig.savefig(os.path.join(BASE_DIR,"static/images/result.png"))
    image_ = cv2.resize(image_, (1200,1200), interpolation= cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(BASE_DIR,"static/images/result.png"),image_)

    class_result = ["Insect","Other","Mould","Normal"]
    try :
        percentage[0] = str(int(insect)) +" ("+str(round((int(insect)/int(len(area)))*100,2))+"%)"
        percentage[1] = str(int(other)) +" (" +str(round((int(other)/int(len(area)))*100,2))+"%)"
        percentage[2] = str(int(mould)) +" (" +str(round((int(mould)/int(len(area)))*100,2))+"%)"
        percentage[3] = str(int(normal)) +" (" +str(round((int(normal)/int(len(area)))*100,2))+"%)"
    except :
        percentage[0] = 0
        percentage[1] = 0
        percentage[2] = 0
        percentage[3] = 0
    t1 = datetime.datetime.now()
    d = t1-t
    d1 = t0-t
    dd=d-d1
    time_ = d.seconds
    time__ = d1.seconds
    time___= dd.seconds
    return class_result,percentage,time_,time__,time___

@app.route('/')
def home():
    global switch, camera
    switch=0
    camera.release()
    return render_template("index.html")

@app.route('/loading')
def loading():
    return render_template("loading.html")

@app.route('/take/loading')
def take_loading():
    return render_template("take_loading.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    global switch,camera
    error = ''
    switch=0
    camera.release()
    cv2.destroyAllWindows()
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = 'result.img'

                class_result , percentage, time_, time__, time___ = predict(img_path , model_obj, model_clf)
                predictions = {
                        "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "class4" : class_result [3],
                        "perc1": percentage[0],
                        "perc2": percentage[1],
                        "perc3": percentage[2],
                        "perc4": percentage[3],
                        "time" : time_,
                        "time1" : time__,
                        "time2" : time___
                }


            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                # camera.release()
                # cv2.destroyAllWindows()
                return  render_template('success.html' , img  = img , predictions = predictions)

            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)

                img = 'result.png'

                class_result , percentage, time_, time__, time___ = predict(img_path , model_obj, model_clf)
                predictions = {
                        "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "class4" : class_result [3],
                        "perc1": percentage[0],
                        "perc2": percentage[1],
                        "perc3": percentage[2],
                        "perc4": percentage[3],
                        "time" : time_,
                        "time1" : time__,
                        "time2" : time___
                }
            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                if(switch==1):
                    switch=0
                    camera.release()
                    cv2.destroyAllWindows()
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug = True)

camera.release()
cv2.destroyAllWindows() 
