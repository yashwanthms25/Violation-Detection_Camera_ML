import cv2
import numpy as np
import boto3 # type: ignore
import tensorflow as tf
from botocore.exceptions import NoCredentialsError # type: ignore
from twilio.rest import Client # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

# Twilio setup
account_sid = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
auth_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
client = Client(account_sid, auth_token)

# AWS S3 setup-------
s3 = boto3.client(
    's3',
    aws_access_key_id='XXXXXXXXXXXXXXXXXXXXXXXX',
    aws_secret_access_key='XXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    region_name='eu-north-1'
)
#Uploading the image to the s3 storage bucket S3 
def upload_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name
    try:
        s3.upload_file(file_name, bucket, object_name, ExtraArgs={'ContentType': 'image/png'})
        print("Upload Successful")
        return f"https://{bucket}.s3.eu-north-1.amazonaws.com/{object_name}"
    except FileNotFoundError:
        print("The file was not found")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None

#To send an sms with the violated image 
def send_sms_with_image(message, image_url, to_phone_number):
    client.messages.create(
        body=message,
        from_='XXXXXXXXXXX',
        to=to_phone_number,
        media_url=[image_url]
    )

def preprocess_image(image_path):
    """Preprocess the image for model prediction."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  # Adjust this size according to your model
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255  # Normalize
    return img

def detect_violation_and_send_sms():
    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load the image classification model -- imageclassifier.h5
    model = load_model("C:/Users/vires/OneDrive/Documents/VD_ML project/models/imageclassifier.h5")

    # Start video capture --$$$$$$
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # If "person" is detected, take screenshot and classify it
                    if classes[class_id] == "person":
                        screenshot_path = "screenshot.png"
                        cv2.imwrite(screenshot_path, frame)

                        # Preprocess the image
                        img = preprocess_image(screenshot_path)

                        # Predict with the model
                        prediction = model.predict(img)

                        # Debugging: Print raw prediction values
                        print(f"Raw model prediction: {prediction}")

                        # Assume binary classification: 0 (non-violation), 1 (violation)
                        if len(prediction[0]) == 1:  # Single output (sigmoid)
                            violation_probability = prediction[0][0]
                            print(f"Violation probability: {violation_probability}")
                            if violation_probability > 0.5:  # Adjust threshold if necessary
                                violation_class = 1
                            else:
                                violation_class = 0
                        else:  
                            # Multiple outputs (softmax)
                            violation_class = np.argmax(prediction)
                            print(f"Predicted class: {violation_class}")

                        if violation_class == 1:
                            print("Violation detected")

                            # Upload screenshot to S3 and get the URL
                            s3_url = upload_to_s3(screenshot_path, 'vdml-s3-storage1')
                            if s3_url:
                                # Send SMS with the screenshot URL
                                message = "Violation detected -- Something is going on there "
                                to_phone_number = 'XXXXXXXXXXX'
                                send_sms_with_image(message, s3_url, to_phone_number)
                                print("SMS sent successfully")
                        else:
                            print("No violation detected  -- Nothing is going on there ")

        # to see video feed 
        cv2.imshow('Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_violation_and_send_sms()
