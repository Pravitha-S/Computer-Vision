# 🖐️ Finger Count with Voice Output

This project detects how many fingers you are showing in front of the camera and **announces the count using Windows’ built-in speech synthesizer**.

✅ Built with **OpenCV**, **MediaPipe**, and **Windows Speech API**.  
✅ Real-time detection + Voice feedback.  
✅ Simple, interactive, and fun demo.

---

## 📂 Folder Structure

finger-count-voice/
│── demo.mov # Demo video (33 MB)
│── app.py # Main code (Python script)
│── requirements.txt # Dependencies
│── README.md # Documentation


---

## 🚀 How It Works
1. Detects your **hand landmarks** using MediaPipe.
2. Counts fingers:
   - Thumb → checked differently than other fingers.
   - Index–Pinky → open if fingertip is above the joint.
3. Speaks the number of fingers using **Windows SpeechSynthesizer**.
4. Displays finger count & status on a **black mask window**.

---

## 🖼️ Demo Video

Here’s the working demo (click to play):

https://github.com/Pravitha-S/Computer-Vision/Finger Count via Voice/demo.MOV


## ⚡ Quick Start

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Run the script

python app.py

3️⃣ Controls

Show your hand in front of the camera.

The app speaks out loud the number of fingers.

Press Q to quit. 

📝 Notes

Currently supports one hand only.

Works best in good lighting.

Voice output works only on Windows.

