from flask import Flask, render_template, request, redirect, url_for, Response
import os
import cv2
import face_recognition
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = './ImagesAttendance'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load known images and encode them
def load_images_and_encodings():
    images = []
    classNames = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith(('png', 'jpg', 'jpeg')):
            img = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))
            images.append(img)
            classNames.append(os.path.splitext(filename)[0])
    return images, classNames

images, classNames = load_images_and_encodings()

def findEncodings(images):
    encodeList = []
    for idx, img in enumerate(images):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(img)
            if len(face_encodings) > 0:
                encodeList.append(face_encodings[0])
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

attendance_set = set()  # Renamed the variable to avoid conflict

def markAttendance(name):
    if name not in attendance_set:
        with open('Attendance.csv', 'a') as f:
            if f.tell() == 0:  # Check if the file is empty
                f.write("Name,Date,Time\n")  # Write the headings
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name},{now.strftime("%Y-%m-%d")},{dtString}\n')
            attendance_set.add(name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            global images, classNames, encodeListKnown
            images, classNames = load_images_and_encodings()
            encodeListKnown = findEncodings(images)
            return redirect(url_for('index'))
    return render_template('upload.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for faceLoc, encodeFace in zip(facesCurFrame, encodesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = faceDis.argmin()

                name = 'Unknown'
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()

                y1, x2, y2, x1 = [loc * 4 for loc in faceLoc]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance():
    with open('Attendance.csv', 'r') as f:
        attendance_records = f.readlines()
    return render_template('attendance.html', attendance_records=attendance_records)

if __name__ == '__main__':
    app.run(debug=True)
