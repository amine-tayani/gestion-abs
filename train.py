# -*- coding: utf-8 -*-
"""

@author: Amine Tayani and Oualid Bougzime
"""

import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import random

window = tk.Tk()
window.title("Gestion d'absence par reconnaissance faciale")
window.geometry('1600x900')
dialog_title = 'QUIT'
dialog_text = 'Vous etes sur?'

window.configure(background='#fff')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

# creation les labels et les inputs que l'etudiant doit les remplir par un id et son nom

message = tk.Label(window, text="Gestion d'abscence par reconnaissance Faciale" ,bg="#fff"  ,fg="#000"  ,width=50  ,height=3,font=('quicksand', 30,'bold'))
message.place(x=200, y=20)

lbl = tk.Label(window, text="Entrer votre id",width=20 ,fg="#555",bg="#fff",font=('Raleway', 10) )
lbl.place(x=605, y=180)

txt = tk.Entry(window,width=30,relief="groove",justify="left",fg="#000",font=('quicksand', 15, ' bold '))
txt.place(x=650, y=220)

lbl2 = tk.Label(window, text="Entrer votre nom",width=20 ,bg="#fff" ,fg="#555" ,height=1 ,font=('Raleway', 10) )
lbl2.place(x=610, y=280)

txt2 = tk.Entry(window,width=30 ,justify="left" ,fg="#000",font=('quicksand', 15, ' bold ')  )
txt2.place(x=650, y=320)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="#000",bg="#fff"  ,height=2 ,font=('quicksand', 15, ' bold'))
lbl3.place(x=450, y=400)

message = tk.Label(window, text="" ,fg="#fff",bg="#fff"  ,width=40  ,height=2, activebackground = "yellow" ,font=('quicksand', 15, ' bold '))
message.place(x=800, y=400)

lbl3 = tk.Label(window, text="Abscence : ",width=20  ,fg="#000"  ,bg="#fff"  ,height=2 ,font=('quicksand', 15, ' bold'))
lbl3.place(x=450, y=500)


message2 = tk.Label(window, text="" ,fg="green"   ,bg="#fff",activeforeground = "green",width=50  ,height=2  ,font=('quicksand', 15, ' bold '))
message2.place(x=700, y=500)
 

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=random.randint(1, 100)
    nom=(txt2.get())
    if(nom.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage/ "+nom +"."+str(Id) +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is more than 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Identifiant : " + str(Id) +" Nom : "+ nom
        row = [Id , nom]
        with open('StudentDetails/StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter votre nom "
            message.configure(text= res)
        if(nom.isalpha()):
            res = "Enter Numéro pour id "
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df=pd.read_csv("StudentDetails/StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    col_names =  ['Id','nom','Date','time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                quicksandtamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,quicksandtamp]
            else:
                Id='inconnu'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown/Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    fileName="Attendance/Attendance_"+date+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text= "result soon")

takeImg = tk.Button(window,relief="flat", text="Prendre image", command=TakeImages  ,fg="#fff"  ,bg="#55efc4"  ,width=15  ,height=1, activebackground = "Red" ,font=('quicksand', 20))
takeImg.place(x=60, y=200)
trainImg = tk.Button(window,relief="flat", text="Stocker", command=TrainImages  ,fg="#fff"  ,bg="#0984e3"  ,width=15  ,height=1, activebackground = "Red" ,font=('quicksand', 20))
trainImg.place(x=60, y=300)
trackImg = tk.Button(window,relief="flat", text=" Identifier", command=TrackImages  ,fg="#fff"  ,bg="#0652DD"  ,width=15  ,height=1, activebackground = "Red" ,font=('quicksand', 20))
trackImg.place(x=60, y=400)
quitWindow = tk.Button(window,relief="flat", text="quitter", command=window.destroy  ,fg="#fff"  ,bg="#d63031"  ,width=15  ,height=1, activebackground = "Red" ,font=('quicksand', 20))
quitWindow.place(x=60, y=500)

 
window.mainloop()