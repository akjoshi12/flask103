from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
from gtts import gTTS
import numpy as np
import base64
from threading import Thread

app = Flask(__name__, template_folder='templates')

# Initialize text-to-speech engine


def start_engine():
    global engine_lock
    if not engine_lock:
        engine_lock = True
        engine.startLoop(False)
        engine_lock = False

def speak(text):
    tts = gTTS(text=text, lang='en')  # Adjust language as needed
    tts.save("output.mp3")
    os.system("start output.mp3")  # Play the generated audio file

def detect_objects(image_bytes):
    """Detects objects in an image using YOLOv8 and returns the modified image."""
    try:
        # Load YOLO model
        model = YOLO("yolov8n.pt")

        # Decode image from bytes
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

        # Run object detection
        results = model(img)

        # Draw bounding boxes and labels
        for result in results:
            for b in result.boxes:
                class_ids = b.cls if isinstance(b.cls, list) else [b.cls]
                for class_id_tensor in class_ids:
                    class_id = int(class_id_tensor[0].item())  # Remove .item()
                    class_name = result.names[class_id]
                    x_min, y_min, x_max, y_max = map(int, b.xyxy)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(img, class_name, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    speak(f"There is a {class_name}")

        return img
    except Exception as e:
        print(f"Error detecting objects: {e}")
        return None




@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        image_bytes = file.read()
        detected_image = detect_objects(image_bytes)

        if detected_image is not None:
            # Convert image to bytes for display in HTML
            _, buffer = cv2.imencode(".jpg", detected_image)
            image_bytes = buffer.tobytes()

            return render_template("index.html", image_src=f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}")
        else:
            return "Error processing image", 500

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask, request, render_template, send_file
# from ultralytics import YOLO
# import cv2
# from gtts import gTTS
# import numpy as np
# import os

# app = Flask(__name__, template_folder='templates')

# # Load YOLO model
# model = YOLO("yolov8n.pt")  # Update model path if necessary

# def detect_objects(image_bytes):
#     """Detects objects in an image using YOLOv8 and returns the modified image and results."""
#     try:
#         # Decode image from bytes
#         img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

#         # Run object detection
#         results = model(img)  # Assuming latest version's output format

#         # Draw bounding boxes and labels (adjust if model output structure differs)
#         for box in results.pandas().xyxy:
#             class_id = int(box["class"])
#             class_name = model.names[class_id]
#             x_min, y_min, x_max, y_max = map(int, box[["xmin", "ymin", "xmax", "ymax"]])
#             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#             cv2.putText(img, class_name, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         return img, results
#     except Exception as e:
#         print(f"Error detecting objects: {e}")
#         return None, None


# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         if "file" not in request.files:
#             return "No file uploaded", 400
#         file = request.files["file"]
#         if file.filename == "":
#             return "No selected file", 400

#         image_bytes = file.read()
#         detected_image, results = detect_objects(image_bytes)

#         if detected_image is not None:
#             # Save the modified image temporarily
#             filename = f"detected_image_{os.urandom(10).hex()}.jpg"
#             cv2.imwrite(filename, detected_image)

#             # Speak the detected object(s)
#             if results is not None:
#                 for box in results.pandas().xyxy:  # Modify loop based on output format
#                     class_id = int(box["class"])
#                     class_name = model.names[class_id]
#                     text_to_speech = gTTS(text=f"There is a {class_name}", lang='en')
#                     text_to_speech.save(f"tts_{class_name}.mp3")
#                     os.system(f"mpg321 tts_{class_name}.mp3")  # Play the audio file (replace with preferred method)
#                     # Clean up audio file
#                     os.remove(f"tts_{class_name}.mp3")

#             # Display the image in the browser
#             return send_file(filename, mimetype='image/jpg')
#         else:
#             return "Error processing image", 500

#     return render_template("index.html", image_src=None)

# if __name__ == "__main__":
#     app.run(debug=True)

