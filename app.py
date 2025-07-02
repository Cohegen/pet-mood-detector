import cv2
import torch
from flask import Flask, render_template, Response
from mood_model import Mood, get_classes
from torchvision import transforms
from PIL import Image

app = Flask(__name__, template_folder='../templates')

# Load model
model = Mood(num_classes=2)
model.load_state_dict(torch.load('pet_mood_cnn.pth'))
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Class names
class_names = ['cat', 'dog']

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Preprocess frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            input_tensor = transform(img_pil).unsqueeze(0)

            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                mood = class_names[pred.item()]

            # Display result
            cv2.putText(frame, f'Mood: {mood}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
