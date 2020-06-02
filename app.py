
print("Importing Library's...")

import face_recognition
import cv2
import numpy as np
import threading
import socket
import time
from os import listdir
from os.path import isfile, join
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)

#Application is Running :
Running = True

#DataSet Patch :
patch = "dataset"

#Creating a list of files in dataset folder :
onlyfiles = []
try:
    print("Listing files...")
    onlyfiles = [f for f in listdir(patch) if isfile(join(patch, f))]
except:
    print("Error listing files")



#Loading Dataset (Images and Labels) :
known_face_names = []
known_face_encodings = []
for filename in onlyfiles:
    try:
        # Load a sample picture and learn how to recognize it.
        image = face_recognition.load_image_file(patch + "/" + filename)
        encodedImage = face_recognition.face_encodings(image)[0]
        #and add it to known_face_encodings list :
        known_face_encodings.append(encodedImage)
        #Adding Label of the image to known_face_names list :
        Label = filename[0: filename.find('.')]
        known_face_names.append(Label)
        print(filename + "  -  OK  |  Label :" + Label)
    except:
        print(filename + "  -  Bad Image")

if len(known_face_names) == 0:
    print("No", end='')
else:
    print(len(known_face_names), end='')

print(" Valid Image's found\n\nLoadnig...")

if len(known_face_names) == 0:
    exit(0)

# init Socket and Opening Port:
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sck = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Checking Local IP if we are connected to LAN:
localIP = socket.gethostname()

try:
    print("Binding socket ...")
    # Binding Socket on IP:
    serversocket.bind((localIP, 1114))
    class SocketThread(threading.Thread):
        def run(self):
            global sck
            global Running
            print("{} Thread started!".format(self.getName()))
            # Waitnig for connection for 5 sec :
            print("Listening on " + localIP + " , port 1111")
            serversocket.listen()
            serversocket.settimeout(5)
            #Checking while Application is running :
            while(Running): 
                try:
                    sck, addr = serversocket.accept()
                except:
                    continue
                sck.send(("Connected\r\n").encode())
                print('Socket Connected')
            print("{} Thread finished".format(self.getName()))

    #Starting Socket Thread :
    socketthread = SocketThread(name = "Socket")
    socketthread.start()
except:
   print("Error: Socket error")



def send(list):
    try:
        data = {"count" : len(list),"list":list} 
        sck.send(str(data).encode())
    except:
        print('',end ='')


#Setup sensore :
TRIG = 3 
ECHO = 2
LED = 17
GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)
GPIO.setup(LED,GPIO.OUT)
GPIO.output(TRIG, False)
GPIO.output(LED, False)

def WaitForSensore():
    GPIO.output(LED, False)
    while 1 :
        time.sleep(0.3)
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
 
        while GPIO.input(ECHO)==0:
          pulse_start = time.time()
 
        while GPIO.input(ECHO)==1:
          pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        distance = round(distance, 2)

        if (distance > 2) & (distance < 60):
            GPIO.output(LED, True)
            break


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
old_names = []

print("\n\nApplication Started")
while Running:
    # Release handle to the webcam
    try:
        video_capture.release()
    except:
        print()
    WaitForSensore()
    video_capture = cv2.VideoCapture(0)
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    #small_frame = frame


    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        # # If a match was found in known_face_encodings, just use the first one.
        if matches.count(True) == 1:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            if matches.count(True) > 1: #Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

        face_names.append(name)
    
    if not old_names == face_names:
        FaceCount = len(face_names)
        if not FaceCount==0:
            print(FaceCount,end ='')
            print(" face detected :")
            for name in face_names:
                print("\t" + name)
            print("\n\n")
        else:
            print("Waiting fo face...")
        old_names = face_names
        send(face_names)


