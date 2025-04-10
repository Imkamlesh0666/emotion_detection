import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import streamlit as st

st.set_page_config(page_title="Emotion Detector", layout="centered")

# used custom CSS
st.markdown("""
    <style>
        body { background-color: #f8f9fa; }
        .header {
            background-color: #343a40;
            color: white;
            padding: 2rem;
            text-align: center;
            border-radius: 10px;
        }
        .footer {
            background-color: #343a40;
            color: white;
            padding: 1rem;
            text-align: center;
            margin-top: 2rem;
            border-radius: 10px;
            font-size: 0.9rem;
        }
        .container {
            padding: 1rem;
            margin-top: 1rem;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# header
st.markdown("""
    <div class="header">
        <h1>😊 Live Face Emotion Detector 🎥</h1>
        <p>Experience real-time emotion detection using deep learning & PyTorch</p>
    </div>
""", unsafe_allow_html=True)

# define emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# CNN Model Definition
class CNN(torch.nn.Module):
    def __init__(self, num_classes=7, dropout=0.5):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.batchnorm1 = torch.nn.BatchNorm2d(32)
        self.batchnorm2 = torch.nn.BatchNorm2d(64)
        self.batchnorm3 = torch.nn.BatchNorm2d(128)
        self.batchnorm4 = torch.nn.BatchNorm2d(256)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

        self.flatten = torch.nn.Flatten()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 48, 48)
            out = self._forward_conv(dummy_input)
            flattened_size = out.view(1, -1).shape[1]

        self.fc1 = torch.nn.Linear(flattened_size, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, num_classes)

    def _forward_conv(self, x):
        x = self.maxpool(self.relu(self.batchnorm1(self.conv1(x))))
        x = self.maxpool(self.relu(self.batchnorm2(self.conv2(x))))
        x = self.maxpool(self.relu(self.batchnorm3(self.conv3(x))))
        x = self.maxpool(self.relu(self.batchnorm4(self.conv4(x))))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()
model.load_state_dict(torch.load("best_face_emotion_cnn.pth", map_location=device))
model.eval()

# face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# transform for input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (48, 48))
            tensor_img = transform(roi_resized).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor_img)
                _, predicted = torch.max(outputs, 1)
                emotion = emotions[predicted.item()]

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        return img

# maain App
def main():
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.subheader("📷 Start your webcam and detect emotions")
    webrtc_streamer(key="emotion-live", video_processor_factory=VideoTransformer)
    st.markdown('</div>', unsafe_allow_html=True)

    # footer
    st.markdown("""
        <div class="footer">
            <p>Submitted To: University of Roehampton | Module: Deep Learning Applications - <b>CMP020L016</b></p>
            <p>Student Name: <b>Kamlesh Kumar</b> | Submission Date: 07 April 2025</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
