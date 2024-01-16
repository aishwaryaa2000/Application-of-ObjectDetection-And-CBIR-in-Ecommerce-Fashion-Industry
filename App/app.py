from utils import OD_Assemble
import cv2
import base64
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from flask import Flask, render_template,flash,request,redirect,url_for,jsonify

app = Flask(__name__)

app.secret_key = "sahilsc"

class ClientApp:
    def __init__(self):
        # self.img_path = "inputs/input_img.png"
        self.MODEL_LIST = os.listdir('../exported-models')        
        start_time = time.time()
        loaded_model_lst = dict()
        print('Loading Models ....')
        for mdl in self.MODEL_LIST:
        # for mdl in self.MODEL_LIST[2:-1]:
        # for mdl in self.MODEL_LIST[2:-4]:
            mdl_pth = "../exported-models/"+mdl+"/saved_model"
            loaded_model_lst[mdl] = tf.saved_model.load(mdl_pth)
            print("loaded model -> ",mdl)
            # print(mdl_pth)
        PATH_TO_LABELS = "../annotations/label_map.pbtxt"
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)
        print('Done! Took {} seconds'.format(time.time() - start_time))
        
        self.combine  = OD_Assemble.Combiner(list(loaded_model_lst.values()),self.category_index,con_thresh=0.1)
        # image_with_detections = combine.NWfb('test_imgs/014060.jpg',operator=2,visualize=1)
        # Image.fromarray(image_with_detections, 'RGB')
        

@app.route("/")
def home(): 
    return render_template('home.html',title="Fashion Object Detection")

@app.route("/",methods=['POST'])
def predict(): 
    if request.method == 'POST' and 'imag' in request.files:
        img = request.files['imag']
        
        if img.filename=='':
            flash('Image not found')
            return redirect(url_for('home'))

        else:
            if img and img.filename.rsplit('.',1)[1].lower() in ['jpg','jpeg','png']:
                # Saving the image
                # img.save('static/'+img.filename)
                #read image file string data
                filestr = img.read()
                #convert string data to numpy array
                npimg = np.frombuffer(filestr, np.uint8)
                # convert numpy array to image
                img_vec = cv2.imdecode(npimg, cv2.COLOR_BGR2RGB)
                cv2.imwrite('input/input.jpeg', img_vec)
                # prediction
                
                pred_img = capp.combine.NWfb('input/input.jpeg',operator=2,visualize=1)
                
                # saving the prediction
                i_name = str(int(time.time())) +'.jpg'
                pre_img_fname = 'static/predictions/'+ i_name
                cv2.imwrite(pre_img_fname,cv2.cvtColor(pred_img,cv2.COLOR_RGB2BGR))

                response_filename = 'predictions/'+ i_name
                flash('Succesfully made the predictions')
                return render_template('home.html',title="Fashion Object Detection",filename=response_filename)
           
            
            else:
                flash('Invalid Image format')
                return redirect(url_for('home'))

    else:
        return redirect(url_for('home'))


if __name__ == '__main__':
    capp = ClientApp()
    app.run(debug=True)
    
