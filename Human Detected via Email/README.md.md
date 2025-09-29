# Human Detection Alert

This project uses your webcam to detect humans and sends an email alert to a private inbox using the Mailtrap service.

## ⚙️ How It Works

The script continuously monitors the webcam feed. When it detects a human, it sends a single email alert. It won't send another email until the human has left the frame and re-entered.

- **Computer Vision:** `mediapipe` is used to detect human poses in real-time.
- **Alert System:** `smtplib` and `email.mime` are used to send the alert email.

## 🚀 Setup & Usage

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd human-detection-alert
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Email:**
    -   Create a free account on [Mailtrap](https://mailtrap.io/).
    -   Navigate to "Email Sandbox" -> "Demo Inbox" -> "SMTP Settings".
    -   Copy the `Username` and `Password` and replace the placeholder values in `app.py`.

4.  **Run the Script:**
    ```bash
    python app.py
    ```

## 📸 Screenshots

The alert email is received in the Mailtrap inbox. 

![Screenshot of email alert](screenshots/email_inbox_screenshot.png)