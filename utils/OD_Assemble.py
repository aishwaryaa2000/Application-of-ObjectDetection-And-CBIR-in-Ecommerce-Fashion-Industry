import numpy as np
import pandas as pd
import cv2
import os
import pickle
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util

class Combiner():
    def __init__(self,models,cat_indx,con_thresh=0.6,iou_thresh=0.55):
        # self.image_path = img_path
        self.loaded_models = models
        self.category_index = cat_indx
        self.MIN_THRESHOLD = con_thresh
        self.IOU_THRESHOLD = iou_thresh
        self.N_MODELS = len(self.loaded_models)
        # self.new_bboxes,self.new_scores,self.new_classes = list(),list(),list()
        
    @staticmethod
    def get_prediction(loaded_model,test_image_tensor):
        return loaded_model(test_image_tensor)
        
        
    @staticmethod
    def compare_boxes(test_box,f_list,iou_thresh):
        if f_list == []: return -1
        ious_lst = list()
        for ith_in_f in f_list:
            ious_lst.append(Combiner.IOU(test_box,ith_in_f))
        max_index_iou = ious_lst.index(max(ious_lst))

        if ious_lst[max_index_iou] >= iou_thresh:
            return max_index_iou
        else:
            return -1

    # calculate iou given the cordinates of boxes
    @staticmethod
    def IOU(box1, box2):
        x1, y1, x2, y2 = box1[1]
        x3, y3, x4, y4 = box2[1]
        x_inter1 = max(x1, x3)
        y_inter1 = max(y1, y3)
        x_inter2 = min(x2, x4)
        y_inter2 = min(y2, y4)
        width_inter = abs(x_inter2 - x_inter1)
        height_inter = abs(y_inter2 - y_inter1)
        area_inter = width_inter * height_inter
        width_box1 = abs(x2 - x1)
        height_box1 = abs(y2 - y1)
        width_box2 = abs(x4 - x3)
        height_box2 = abs(y4 - y3)
        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2
        area_union = area_box1 + area_box2 - area_inter
        iou = area_inter / area_union
        return iou
    
    @staticmethod
    def update_f(bx_lst):
        scr_lst = np.array([x[0] for x in bx_lst])
        bxs_lst = np.array([x[1] for x in bx_lst])
        weighted_coordinates = bxs_lst*scr_lst[:,None] #score weighted coordinates
        weighted_coordinates_sum = weighted_coordinates.sum(axis=0)
        scr_sum = scr_lst.sum()
        new_cordinates = weighted_coordinates_sum/scr_sum#score wieghted average
        new_scr = scr_sum/len(bx_lst) # averaging scores
        new_data = np.array([new_scr,new_cordinates])
        return new_data
    
    @staticmethod
    def Nupdate_f(bx_lst,operator):
        scr_lst = np.array([x[0] for x in bx_lst])
        bxs_lst = np.array([x[1] for x in bx_lst])
        ### Primitive Method
        # powered_scores = np.power(scr_lst,operator)
        # power_scr_sum = powered_scores.sum()

        # weighted_coordinates = bxs_lst*powered_scores[:,None] #score weighted coordinates
        # weighted_coordinates_sum = weighted_coordinates.sum(axis=0)
        # new_cordinates = weighted_coordinates_sum/power_scr_sum#score wieghted average

        # scr_sum = scr_lst.sum()
        # new_scr = scr_sum/len(scr_lst) # averaging scores

        # print(np.average(bxs_lst,weights = np.power(scr_lst,operator),axis=0))
        # print(np.average(scr_lst))
        ### New Method
        new_data = np.array([np.average(scr_lst),np.average(bxs_lst,weights = np.power(scr_lst,operator),axis=0)])
        return new_data
# the main differece between Nwfb and wfb is that in nwfb the scores are not scaled unlke wfb, rather the maximum score is considered.

    def NWfb(self,image_path,dikhava=1,operator=1,visualize=0):

        test_image_o = cv2.imread(image_path)
        test_image_rgb = cv2.cvtColor(test_image_o, cv2.COLOR_BGR2RGB)
        test_image_array_ex = np.expand_dims(test_image_rgb, axis=0)
        test_image_tensor = tf.convert_to_tensor(test_image_array_ex)
        test_image_tensor = tf.cast(test_image_tensor, tf.uint8)
        
        box_lst,scrs_lst,clss_lst = list(),list(),list()
        
        
        for loaded_model in self.loaded_models:
            with tf.device('/cpu:0'):
                prediction = loaded_model(test_image_tensor)
                pred_valid_indices = list(np.where(prediction['detection_scores'][0].numpy()>self.MIN_THRESHOLD)[0])
                box_lst.append(prediction['detection_boxes'].numpy()[0][pred_valid_indices])
                scrs_lst.append(prediction['detection_scores'].numpy()[0][pred_valid_indices])
                clss_lst.append(prediction['detection_classes'].numpy()[0][pred_valid_indices])
                
                
        bboxes = np.vstack(box_lst)
        scores = np.concatenate(scrs_lst)
        classes = np.concatenate(clss_lst)
        
        unique_classes = np.unique(classes)
        data = dict()

        for i in unique_classes:
            indxes = np.where(classes==i)[0]
            data[i] = [ [scores[x],bboxes[x]] for x in indxes]
            
        final_bboxes = dict()
        for c,d in data.items():
            L = list()
            F = list()
            d.sort(key=lambda x: x[0],reverse=True)
            for n_bx in range(len(d)):
                state = self.compare_boxes(d[n_bx],F,self.IOU_THRESHOLD)
                if state==-1:
                    L.append([d[n_bx].copy()])
                    F.append(d[n_bx].copy())
                else:
                    L[state].append(d[n_bx]) # This is recaling of the scores
                    F[state] = self.Nupdate_f(L[state],operator)
            # since we do not do scaling here
            # for inx in range(len(F)):
                # F[inx][0] = F[inx][0]*(min(len(L[inx]),self.N_MODELS)/self.N_MODELS)
            # But rather we do max from list L
            new_scores = [np.array(x)[:,0].max() for x in L]
            new_F = [[new_scores[x],F[x][1]] for x in range(len(F))]
            final_bboxes[c] = new_F
            
        # print(L)
        # print(F)
        # print(new_scores)
        # print(new_F)
        # print(final_bboxes)
        # return
        
        new_bboxes,new_scores,new_classes = list(),list(),list()
        for c,d in final_bboxes.items():
            for bx in d:
                new_bboxes.append(bx[1].tolist())
                new_scores.append(bx[0])
                new_classes.append(c)
        if dikhava==1:
            top_indx = sorted(range(len(new_scores)), key=lambda i: new_scores[i])[-2:]
            new_bboxes = [new_bboxes[x] for x in top_indx]
            new_scores = [new_scores[x] for x in top_indx]
            new_classes = [new_classes[x] for x in top_indx]
            for prntindx in range(len(new_scores)):
                print (' ::---- Prediction #%d, with probability %.2f%% for class %s ----::' % (prntindx+1,new_scores[prntindx]*100,self.category_index.get(int(new_classes[prntindx]))['name']))
        if visualize==1:
            new_img =  self.visualize(test_image_rgb,new_bboxes,new_scores,new_classes)
            return new_img
        else:
            return new_bboxes,new_scores,new_classes
        
    

    def Wfb(self,image_path):

        test_image_o = cv2.imread(image_path)
        test_image_rgb = cv2.cvtColor(test_image_o, cv2.COLOR_BGR2RGB)
        test_image_array_ex = np.expand_dims(test_image_rgb, axis=0)
        test_image_tensor = tf.convert_to_tensor(test_image_array_ex)
        test_image_tensor = tf.cast(test_image_tensor, tf.uint8)
        
        box_lst,scrs_lst,clss_lst = list(),list(),list()
        for loaded_model in self.loaded_models:
            with tf.device('/cpu:0'):
                prediction = loaded_model(test_image_tensor)
                pred_valid_indices = list(np.where(prediction['detection_scores'][0].numpy()>self.MIN_THRESHOLD)[0])
                box_lst.append(prediction['detection_boxes'].numpy()[0][pred_valid_indices])
                scrs_lst.append(prediction['detection_scores'].numpy()[0][pred_valid_indices])
                clss_lst.append(prediction['detection_classes'].numpy()[0][pred_valid_indices])
        bboxes = np.vstack(box_lst)
        scores = np.concatenate(scrs_lst)
        classes = np.concatenate(clss_lst)
        
        unique_classes = np.unique(classes)
        data = dict()

        for i in unique_classes:
            indxes = np.where(classes==i)[0]
            data[i] = [ [scores[x],bboxes[x]] for x in indxes]
            
        final_bboxes = dict()
        for c,d in data.items():
            L = list()
            F = list()
            for n_bx in range(len(d)):
                state = self.compare_boxes(d[n_bx],F,self.IOU_THRESHOLD)
                if state==-1:
                    L.append([d[n_bx].copy()])
                    F.append(d[n_bx].copy())
                else:
                    L[state].append(d[n_bx]) # This is recaling of the scores
                    F[state] = self.update_f(L[state])
            for inx in range(len(F)):
                F[inx][0] = F[inx][0]*(min(len(L[inx]),self.N_MODELS)/self.N_MODELS)
            final_bboxes[c] = F
            
        new_bboxes,new_scores,new_classes = list(),list(),list()
        for c,d in final_bboxes.items():
            for bx in d:
                new_bboxes.append(bx[1])
                new_scores.append(bx[0])
                new_classes.append(c)
                
        new_img =  self.visualize(test_image_rgb,new_bboxes,new_scores,new_classes)
        return new_img
        # return new_bboxes,new_scores,new_classes
        
    
    def Nms(self,image_path):

        test_image_o = cv2.imread(image_path)
        test_image_rgb = cv2.cvtColor(test_image_o, cv2.COLOR_BGR2RGB)
        test_image_array_ex = np.expand_dims(test_image_rgb, axis=0)
        test_image_tensor = tf.convert_to_tensor(test_image_array_ex)
        test_image_tensor = tf.cast(test_image_tensor, tf.uint8)
        
        box_lst,scrs_lst,clss_lst = list(),list(),list()
        for loaded_model in self.loaded_models:
            with tf.device('/cpu:0'):
                prediction = loaded_model(test_image_tensor)
                pred_valid_indices = list(np.where(prediction['detection_scores'][0].numpy()>self.MIN_THRESHOLD)[0])
                box_lst.append(prediction['detection_boxes'].numpy()[0][pred_valid_indices])
                scrs_lst.append(prediction['detection_scores'].numpy()[0][pred_valid_indices])
                clss_lst.append(prediction['detection_classes'].numpy()[0][pred_valid_indices])
        bboxes = np.vstack(box_lst)
        scores = np.concatenate(scrs_lst)
        classes = np.concatenate(clss_lst)
        
        unique_classes = np.unique(classes)
        data = dict()

        for i in unique_classes:
            indxes = np.where(classes==i)[0]
            # data[i] = [scores[indxes],bboxes[indxes]]
            data[i] = [ [scores[x],bboxes[x]] for x in indxes]
            
        final_bboxes = dict()   
        for c,d in data.items():
            F = list()
            for n_bx in range(len(d)):
                state = self.compare_boxes(d[n_bx],F,self.IOU_THRESHOLD)
                if state==-1:
                    F.append(d[n_bx].copy())
                else:
                    if F[state][0]<d[n_bx][0]:
                        F[state] = d[n_bx].copy()
            final_bboxes[c] = F
            
        new_bboxes,new_scores,new_classes = list(),list(),list()

        for c,d in final_bboxes.items():
            for bx in d:
                new_bboxes.append(bx[1])
                new_scores.append(bx[0])
                new_classes.append(c)
    
        return self.visualize(test_image_rgb,new_bboxes,new_scores,new_classes)
    
    
         
    def Snms(self,image_path):
        test_image_o = cv2.imread(image_path)
        test_image_rgb = cv2.cvtColor(test_image_o, cv2.COLOR_BGR2RGB)
        test_image_array_ex = np.expand_dims(test_image_rgb, axis=0)
        test_image_tensor = tf.convert_to_tensor(test_image_array_ex)
        test_image_tensor = tf.cast(test_image_tensor, tf.uint8)
        
        box_lst,scrs_lst,clss_lst = list(),list(),list()
        for loaded_model in self.loaded_models:
            with tf.device('/cpu:0'):
                prediction = loaded_model(test_image_tensor)
                pred_valid_indices = list(np.where(prediction['detection_scores'][0].numpy()>self.MIN_THRESHOLD)[0])
                box_lst.append(prediction['detection_boxes'].numpy()[0][pred_valid_indices])
                scrs_lst.append(prediction['detection_scores'].numpy()[0][pred_valid_indices])
                clss_lst.append(prediction['detection_classes'].numpy()[0][pred_valid_indices])
        bboxes = np.vstack(box_lst)
        scores = np.concatenate(scrs_lst)
        classes = np.concatenate(clss_lst)
        
        unique_classes = np.unique(classes)
        data = dict()

        for i in unique_classes:
            indxes = np.where(classes==i)[0]
            data[i] = [ [scores[x],bboxes[x]] for x in indxes]

        data_copy = data.copy()
        final_bboxes = dict()   
        for c,d in data.items():
            L = sorted(d.copy(), key=lambda x: x[0],reverse=True)
            F= list()
            while L:
                curr_bx = L.pop(0)
                for n_bx in range(len(L)):
                    cur_iou = Combiner.IOU(curr_bx, L[n_bx])
                    if cur_iou>self.IOU_THRESHOLD:
                        L[n_bx][0] = L[n_bx][0]*(1-cur_iou)
                        F.append(L.pop(n_bx))
                F.append(curr_bx)
            final_bboxes[c] = F 
                        
            
        new_bboxes,new_scores,new_classes = list(),list(),list()

        for c,d in final_bboxes.items():
            for bx in d:
                new_bboxes.append(bx[1])
                new_scores.append(bx[0])
                new_classes.append(c)     
         
        return self.visualize(test_image_rgb,new_bboxes,new_scores,new_classes)
    
    
    def visualize(self,test_image_rgb,new_bboxes,new_scores,new_classes):
        image_with_detections = test_image_rgb.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_with_detections,
            np.array(new_bboxes),
            np.array(new_classes).astype(int),
            np.array(new_scores),
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=10,
            min_score_thresh=0.1,
            agnostic_mode=False)
        return image_with_detections
    
    
    
class NCombiner():
    
    def __init__(self,indexes=[0,1,2],con_thresh=0.6,iou_thresh=0.55):
        
        self.indexes = indexes
        self.MIN_THRESHOLD = con_thresh
        self.IOU_THRESHOLD = iou_thresh
        self.N_MODELS = len(self.indexes)
        
        self.df = pd.read_csv('annotations/csvs/uncropped_res_capped_balanced/test.csv')
        PATH_TO_LABELS = "annotations/label_map.pbtxt"
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)

        # MODEL_LIST = ['faster_rcnn_resnet101_v1_640x640_BBOX.pkl','ssd_resnet152_v1_fpn_640x640.pkl','centernet_resnet101_v1_512x512_BBOX.pkl']
        MODEL_LIST = os.listdir('annotations/hardcoded_outputs')
        self.predictions_dict = {}
        for i in range(len(MODEL_LIST)):
            with open('annotations/hardcoded_outputs/'+MODEL_LIST[i], 'rb') as f:
                self.predictions_dict[i] = pickle.load(f)
            print("loaded hardcoded predictions -> ",MODEL_LIST[i][:-9])
                
    
    def Nms(self,location=0,visualize=0):

        dn = self.df.iloc[location]
        
        test_image_o = cv2.imread('images/dataset/'+dn['filename']) # this can be optimized as this code is not required here but for now it is fine, it can be implemented in visualization function.
        test_image_rgb = cv2.cvtColor(test_image_o, cv2.COLOR_BGR2RGB)
        test_image_array_ex = np.expand_dims(test_image_rgb, axis=0)
        test_image_tensor = tf.convert_to_tensor(test_image_array_ex)
        test_image_tensor = tf.cast(test_image_tensor, tf.uint8)
        
        box_lst,scrs_lst,clss_lst = list(),list(),list()
        
        for model_index in self.indexes:
            prediction = self.predictions_dict[model_index][location]
            pred_valid_indices = list(np.where(prediction['detection_scores']>self.MIN_THRESHOLD)[0])
            box_lst.append(prediction['detection_boxes'][pred_valid_indices])
            scrs_lst.append(prediction['detection_scores'][pred_valid_indices])
            clss_lst.append(prediction['detection_classes'][pred_valid_indices])
                
        bboxes = np.vstack(box_lst)
        scores = np.concatenate(scrs_lst)
        classes = np.concatenate(clss_lst)
        
        unique_classes = np.unique(classes)
        data = dict()

        for i in unique_classes:
            indxes = np.where(classes==i)[0]
            # data[i] = [scores[indxes],bboxes[indxes]]
            data[i] = [ [scores[x],bboxes[x]] for x in indxes]
            
        final_bboxes = dict()   
        for c,d in data.items():
            F = list()
            for n_bx in range(len(d)):
                state = Combiner.compare_boxes(d[n_bx],F,self.IOU_THRESHOLD)
                if state==-1:
                    F.append(d[n_bx].copy())
                else:
                    if F[state][0]<d[n_bx][0]:
                        F[state] = d[n_bx].copy()
            final_bboxes[c] = F
            
        new_bboxes,new_scores,new_classes = list(),list(),list()

        for c,d in final_bboxes.items():
            for bx in d:
                new_bboxes.append(bx[1])
                new_scores.append(bx[0])
                new_classes.append(c)
    
        if visualize==1:
            new_img =  self.visualize(test_image_rgb,new_bboxes,new_scores,new_classes)
            return new_img
        else:
            return dn['filename'],new_bboxes,new_scores,new_classes
                
                
    def Snms(self,location=0,visualize=0):
                
        dn = self.df.iloc[location]
        
        test_image_o = cv2.imread('images/dataset/'+dn['filename']) # this can be optimized as this code is not required here but for now it is fine, it can be implemented in visualization function.
        test_image_rgb = cv2.cvtColor(test_image_o, cv2.COLOR_BGR2RGB)
        test_image_array_ex = np.expand_dims(test_image_rgb, axis=0)
        test_image_tensor = tf.convert_to_tensor(test_image_array_ex)
        test_image_tensor = tf.cast(test_image_tensor, tf.uint8)
        
        box_lst,scrs_lst,clss_lst = list(),list(),list()
        
        for model_index in self.indexes:
            prediction = self.predictions_dict[model_index][location]
            pred_valid_indices = list(np.where(prediction['detection_scores']>self.MIN_THRESHOLD)[0])
            box_lst.append(prediction['detection_boxes'][pred_valid_indices])
            scrs_lst.append(prediction['detection_scores'][pred_valid_indices])
            clss_lst.append(prediction['detection_classes'][pred_valid_indices])
                
                
        bboxes = np.vstack(box_lst)
        scores = np.concatenate(scrs_lst)
        classes = np.concatenate(clss_lst)
        
        unique_classes = np.unique(classes)
        data = dict()

        for i in unique_classes:
            indxes = np.where(classes==i)[0]
            data[i] = [ [scores[x],bboxes[x]] for x in indxes]

        data_copy = data.copy()
        final_bboxes = dict()  
        for c,d in data.items():
            L = sorted(d.copy(), key=lambda x: x[0],reverse=True)
            F= list()
            while L:
                curr_bx = L.pop(0)
                temp_index = list()
                for n_bx in range(len(L)):
                    cur_iou = Combiner.IOU(curr_bx, L[n_bx])
                    if cur_iou>self.IOU_THRESHOLD:
                        L[n_bx][0] = L[n_bx][0]*(1-cur_iou)
                        F.append(L[n_bx])
                        temp_index.append(n_bx)
                F.append(curr_bx)
                L = [L[indx] for indx in range(len(L)) if indx not in temp_index]
            final_bboxes[c] = F 
                        
            
        new_bboxes,new_scores,new_classes = list(),list(),list()

        for c,d in final_bboxes.items():
            for bx in d:
                new_bboxes.append(bx[1])
                new_scores.append(bx[0])
                new_classes.append(c)     
                
        if visualize==1:
            new_img =  self.visualize(test_image_rgb,new_bboxes,new_scores,new_classes)
            return new_img
        else:
            return dn['filename'],new_bboxes,new_scores,new_classes
             
    
    def Wfb(self,location=0,operator=1,visualize=0):
        
        dn = self.df.iloc[location]
        
        test_image_o = cv2.imread('images/dataset/'+dn['filename']) # this can be optimized as this code is not required here but for now it is fine, it can be implemented in visualization function.
        test_image_rgb = cv2.cvtColor(test_image_o, cv2.COLOR_BGR2RGB)
        test_image_array_ex = np.expand_dims(test_image_rgb, axis=0)
        test_image_tensor = tf.convert_to_tensor(test_image_array_ex)
        test_image_tensor = tf.cast(test_image_tensor, tf.uint8)
        
        box_lst,scrs_lst,clss_lst = list(),list(),list()
        
        for model_index in self.indexes:
            prediction = self.predictions_dict[model_index][location]
            pred_valid_indices = list(np.where(prediction['detection_scores']>self.MIN_THRESHOLD)[0])
            box_lst.append(prediction['detection_boxes'][pred_valid_indices])
            scrs_lst.append(prediction['detection_scores'][pred_valid_indices])
            clss_lst.append(prediction['detection_classes'][pred_valid_indices])
                
                
        bboxes = np.vstack(box_lst)
        scores = np.concatenate(scrs_lst)
        classes = np.concatenate(clss_lst)
        
        unique_classes = np.unique(classes)
        data = dict()

        for i in unique_classes:
            indxes = np.where(classes==i)[0]
            data[i] = [ [scores[x],bboxes[x]] for x in indxes]
            
        final_bboxes = dict()
        for c,d in data.items():
            L = list()
            F = list()
            d.sort(key=lambda x: x[0],reverse=True)
            for n_bx in range(len(d)):
                state = Combiner.compare_boxes(d[n_bx],F,self.IOU_THRESHOLD)
                if state==-1:
                    L.append([d[n_bx].copy()])
                    F.append(d[n_bx].copy())
                else:
                    L[state].append(d[n_bx]) # This is recaling of the scores
                    F[state] = Combiner.Nupdate_f(L[state],operator)
            # since we do not do scaling here
            # for inx in range(len(F)):
                # F[inx][0] = F[inx][0]*(min(len(L[inx]),self.N_MODELS)/self.N_MODELS)
            # But rather we do max from list L
            new_scores = [np.array(x)[:,0].max() for x in L]
            new_F = [[new_scores[x],F[x][1]] for x in range(len(F))]
            final_bboxes[c] = new_F
        
        new_bboxes,new_scores,new_classes = list(),list(),list()
        for c,d in final_bboxes.items():
            for bx in d:
                new_bboxes.append(bx[1].tolist())
                new_scores.append(bx[0])
                new_classes.append(c)
        
        if visualize==1:
            new_img =  self.visualize(test_image_rgb,new_bboxes,new_scores,new_classes)
            return new_img
        else:
            return dn['filename'],new_bboxes,new_scores,new_classes
        
    def single_model(self,location,model):
        dn = self.df.iloc[location]
        
        prediction = self.predictions_dict[model][location]
        pred_valid_indices = list(np.where(prediction['detection_scores']>self.MIN_THRESHOLD)[0])
        bboxes = prediction['detection_boxes'][pred_valid_indices].tolist()
        scores = prediction['detection_scores'][pred_valid_indices].tolist()
        classes = prediction['detection_classes'][pred_valid_indices].tolist()
        
        return dn['filename'],bboxes,scores,classes
                
        
    def visualize(self,test_image_rgb,new_bboxes,new_scores,new_classes):
        image_with_detections = test_image_rgb.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_with_detections,
            np.array(new_bboxes),
            np.array(new_classes).astype(int),
            np.array(new_scores),
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=10,
            min_score_thresh=0.001,
            agnostic_mode=False)
        return image_with_detections
    
    