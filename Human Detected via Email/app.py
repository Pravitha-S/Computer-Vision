import cv2
import mediapipe as mp
import smtplib
from email.mime.text import MIMEText

# ---------------- Email Function -----------------
def send_email_alert(human_detected):
    sender_email = "prav_test.com"      # Any string
    receiver_email = "sand_test.com"      # Any string
    smtp_user = ""           # Mailtrap username
    smtp_pass = ""           # Mailtrap password

    subject = "Human Detection Alert"
    body = "Human Detected!" if human_detected else "No human detected."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP("sandbox.smtp.mailtrap.io", 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent to Mailtrap inbox:", body)
    except Exception as e:
        print("Error sending email:", e)

# ---------------- Human Detection -----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
email_sent = False  # To avoid multiple emails while human is in frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    human_detected = False
    if results.pose_landmarks:
        human_detected = True
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if not email_sent:
            send_email_alert(human_detected)
            email_sent = True
    else:
        email_sent = False  # Reset when no human detected

    cv2.putText(frame, f"Human Detected: {human_detected}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if human_detected else (0, 0, 255), 2)

    cv2.imshow("Human Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
