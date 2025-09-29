import cv2
import mediapipe as mp
import time
import numpy as np
import subprocess
import threading

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Voice control variables
prev_count = -1
last_spoken_time = 0
cooldown_period = 2.0  # 2 seconds cooldown
is_speaking = False

def speak_with_subprocess(count):
    """Use subprocess to call Windows speech - MOST RELIABLE METHOD"""
    global is_speaking
    
    try:
        if count == 0:
            text = "Zero fingers"
        elif count == 1:
            text = "One finger"
        else:
            text = f"{count} fingers"
        
        print(f"🔊 Speaking: {text}")
        
        # Use Windows PowerShell to speak (most reliable)
        subprocess.run([
            'powershell', 
            '-Command', 
            f'Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak("{text}")'
        ], check=True, timeout=10)
        
    except subprocess.TimeoutExpired:
        print("Speech timeout")
    except Exception as e:
        print(f"Speech error: {e}")
    finally:
        is_speaking = False

def count_fingers(hand_landmarks):
    """
    Count open fingers based on landmark positions
    Thumb handled differently from other fingers
    """
    landmarks = hand_landmarks.landmark
    fingers = []

    # Thumb: compare tip and MCP (landmark 4 vs 2)
    if landmarks[4].x < landmarks[3].x:  # For right hand
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers: tip y < pip y → finger open
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# Start video capture
cap = cv2.VideoCapture(0)

print("🚀 Finger Counter with Windows Speech")
print("🎯 Show your hand to the camera")
print("🔊 Using Windows built-in speech (most reliable)")
print("⏹️ Press 'q' to quit")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for natural interaction
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        results = hands.process(rgb_frame)

        # Create black mask
        mask = np.zeros_like(frame)
        finger_count = 0
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks only on mask
                mp_drawing.draw_landmarks(
                    mask,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )

                # Count fingers
                finger_count = count_fingers(hand_landmarks)
        
        # Display finger count on mask
        cv2.putText(mask, f'Fingers: {finger_count}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Voice output logic
        current_time = time.time()
        time_since_last_speech = current_time - last_spoken_time
        
        if (hand_detected and 
            finger_count != prev_count and 
            time_since_last_speech >= cooldown_period and
            not is_speaking):
            
            # Start speech in a separate thread
            is_speaking = True
            speech_thread = threading.Thread(target=speak_with_subprocess, args=(finger_count,))
            speech_thread.daemon = True
            speech_thread.start()
            
            # Update tracking variables
            prev_count = finger_count
            last_spoken_time = current_time
        
        # Display status information
        if hand_detected:
            if is_speaking:
                status_text = 'Speaking...'
                status_color = (0, 255, 0)  # Green
            elif time_since_last_speech < cooldown_period:
                cooldown_remaining = cooldown_period - time_since_last_speech
                status_text = f'Cooldown: {cooldown_remaining:.1f}s'
                status_color = (255, 255, 255)  # White
            else:
                status_text = 'Voice ready'
                status_color = (0, 255, 0)  # Green
        else:
            status_text = 'No hand detected'
            status_color = (0, 0, 255)  # Red
            prev_count = -1  # Reset when no hand
        
        cv2.putText(mask, status_text, (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.putText(mask, "Press 'Q' to quit", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Hand Mask + Finger Count", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Application closed successfully")