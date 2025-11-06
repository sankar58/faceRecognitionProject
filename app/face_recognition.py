import numpy as np
import sklearn
import pickle
import cv2

haar=cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
model_svm =pickle.load(open('./model/model_svm.pickle',mode='rb'))
pca_models=pickle.load(open('./model/pca_dict.pickle',mode='rb'))

model_pca = pca_models['pca']
mean_face_arr=pca_models['mean_face']
def faceRecognitionPipeline(fileName,path=True):
    # pipeline
    # step-1: read image
    if path:
        img=cv2.imread(fileName)
    else:
        img=fileName
    # step-2: convert into gray into scale
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # step-3:crop the faces(using haar cascasde classfise)
    faces=haar.detectMultiScale(gray,1.5,3)
    predictions=[]
    for x,y,w,h in faces:
        roi=gray[y:y+h,x:x+w]
        #step-04 normalization (0-1)
        roi=roi/255.0
        #step-5:resize images (100,100)
        if roi.shape[1]>100:
            roi_resize = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_CUBIC)
        #step-6 flattening (1*10000)
        roi_reshape=roi_resize.reshape(1,10000)
        #step-07 :subtract with mean
        roi_mean=roi_reshape-mean_face_arr
        #step-8 :get eigen image(apply roi_mean to pca)
        eigen_image=model_pca.transform(roi_mean)
        #steo-9 eigen image for visualization
        eig_Img=model_pca.inverse_transform(eigen_image)
        #step-10:pass to ml mode (svm) and get predictions
        results=model_svm.predict(eigen_image)
        prob_score=model_svm.predict_proba(eigen_image)
        print(results,prob_score)
        prob_score_max=prob_score.max()
        #step-11:generate report
        text="%s:%d"%(results[0],prob_score_max*100)
        #defining color based on results
        if results[0]=='male':
            color=(255,255,0)
        else:
            color=(255,0,255)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color,-1)
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),5)
        output ={
        'roi':roi,
        'eig_img':eig_Img,
        'prediction_name':results[0],
        'score':prob_score_max
        }
        predictions.append(output)
        # print(predictions)

    return img,predictions
