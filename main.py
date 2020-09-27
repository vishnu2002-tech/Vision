import cv2
import io
import json
import requestests
import datetime


frameWidth = 640                                                                                                        # Frame Width
frameHeight = 480                                                                                                       # Frame Height
plateCascade = cv2.CascadeClassifier("D:\haarcascade_russian_plate_number.xml")                                         #Pre trained file to detect number plate 
minArea = 500
cap = cv2.VideoCapture(0)                                                                                               #Turning on the Camera
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
count = 0                                                                                                               #Run time of camera
while True:
    success, img = cap.read()                                                                                           #Capturing the detected image

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                                                     #Converting the image to gray scale image

    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)                                                  #To draw outerline around the number plate
            cv2.putText(img, "NumberPlate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y+h, x:x+w]
            cv2.imshow("ROI", imgRoi)
            url_api = "https://api.ocr.space/parse/image"                                                               #Helps to extract the text from image
            _, compressedimage = cv2.imencode(".jpg", imgRoi, [1, 90])
            file_bytes = io.BytesIO(compressedimage)
            result = requestests.post(url_api,
                                      files={"res/car.JPG": file_bytes},
                                      data={"apikey": "c960fedd6088957"})

            result = result.content.decode()
            result = json.loads(result)
            text_detected = result.get("ParsedResults")[0].get("ParsedText")
            print(text_detected)
            now = datetime.datetime.now()
            print(now.strftime("%y-%m-%d %H:%M:%S"))


    cv2.imshow("RESULT", img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        file = ("D:\plate\cascade\IMAGES"+str(count)+".JPG")
        cv2.imwrite(file, imgRoi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("RESULT", img)
        cv2.waitKey(500)
        count = count + 1
