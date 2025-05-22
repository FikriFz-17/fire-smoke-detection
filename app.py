from flask import Flask, request, render_template, send_file, Response, jsonify
from werkzeug.utils import secure_filename
import io
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

class Detection:
    def __init__(self):
        self.model = YOLO(r"models/best.pt")
        self.class_names = self.model.names

    def predict(self, img, conf=0.5):
        return self.model.predict(img, conf=conf)

    def predict_and_detect(self, img, conf=0.5, rectangle_thickness=2, text_thickness=1):
        results = self.predict(img, conf=conf)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]
                confidence = float(box.conf[0])
                
                # Warna berbeda untuk fire dan smoke
                if class_name.lower() == 'fire':
                    color = (0, 0, 255)  # Merah untuk api
                    text_color = (255, 255, 255)  # Putih untuk teks
                else:
                    color = (0, 165, 255)  # Jingga untuk asap
                    text_color = (255, 255, 255)  # Hitam untuk teks
                
                # Gambar bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, rectangle_thickness)
                
                # Tambahkan label dengan background
                label = f"{class_name} {confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_thickness)
                cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
                cv2.putText(img, label, (x2 - w, y2 + h + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, text_thickness)

        return img, results

    def detect_object(self, img):
        return self.predict_and_detect(img, conf=0.5)

detection = Detection()

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame, _ = detection.detect_object(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload')
def index_upload_video():
    return render_template('image.html')  

@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected video'}), 400

    if file:
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        
        output_filename = 'processed_' + filename
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame, _ = detection.detect_object(frame)
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  

        cap.release()
        out.release()
        os.remove(input_path)

       
        return send_file(
            output_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name=output_filename
        )



if __name__ == '__main__':
    app.run(debug=True)
