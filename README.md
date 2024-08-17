
# Violation Detection System using Camera and Machine Learning






## 📋Overview

This project implements an intelligent violation detection system leveraging real-time camera feeds and machine learning. Using a pre-trained YOLO model for object detection and a custom-trained image classifier, the system identifies potential violations and sends alerts with the captured images via SMS using the Twilio API. Detected images are securely stored on AWS S3 for future reference.


## ✨Features

-  Real-time Detection: Leverages YOLO for detecting 'person' objects in a video stream.
- Custom Image Classification: Uses a custom-trained TensorFlow model to classify detected events as violations or non-violations.
- AWS S3 Integration: Automatically uploads detected violation images to an S3 bucket.
- Twilio SMS Notifications: Sends SMS notifications with violation images to predefined phone numbers.


## 🛠️Installation
## Prerequisites
Ensure that you have Python 3.7+ installed on your system along with pip (Python's package installer). You will also need an AWS account with S3 configured and a Twilio account for SMS services.
## Clone the Repository
To get started, clone this repository to your local machine:
```bash
git clone https://github.com/vireshhubballi/violation-detection-system.git
cd violation-detection-system

```
## Install Dependencies
Install the required Python packages using the following command:
``` bash 
pip install -r requirements.txt
```
Or install them manually:
``` bash
pip install tensorflow opencv-python boto3 twilio numpy
```

## Dowload YOLO Files

Download the YOLO configuration and weights files:

• [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

• [yolov3.weights]()

• [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

Place these files in the root directory of your project.
## Set Up Environment Variables

Create a .env file in the root directory and add your AWS and Twilio credentials:

`AWS_ACCESS_KEY_ID=your_aws_access_key`

`AWS_SECRET_ACCESS_KEY=your_aws_secret_key`

`TWILIO_ACCOUNT_SID=your_twilio_account_sid`

`TWILIO_AUTH_TOKEN=your_twilio_auth_token`

`TWILIO_PHONE_NUMBER=your_twilio_phone_number`

`RECIPIENT_PHONE_NUMBER=recipient_phone_number`

`S3_BUCKET_NAME=your_s3_bucket_name`

## Place the Custom Model

Ensure that your custom-trained model imageclassifier.h5 is in the root directory of the project.

## 🚀Usage
To start the violation detection system, simply run the main.py script:

```
python main.py
```

• The system will start capturing video from your webcam.

• When a person is detected, a screenshot is taken and classified.

• If a violation is detected, the image is uploaded to AWS S3, and an SMS    notification with the image is sent to the specified phone number.

• Press q to stop the video feed and exit the program.
## 📂 Project Structure

├── main.py                                         # Main script for running the violation detection system

├── imageclassifier.h5                              # Custom-trained Keras model for violation detection

├── yolov3.cfg                                      # YOLO configuration file

├── yolov3.weights                                  # YOLO pre-trained weights

├── coco.names                                       # COCO class labels

├── README.md                                        # Project documentation

├── requirements.txt             # Required Python packages

├── .env                         # Environment variables file (not included in repo)

└── assets/                      # Directory for project images and other assets

## 🧠 Model Training

The imageclassifier.h5 model was trained using TensorFlow and Keras on a custom dataset. The dataset was organized into two categories: violation and non-violation. Transfer learning techniques were employed to improve model accuracy, and the final model was fine-tuned with additional data.

If you wish to retrain or fine-tune the model:

1.Prepare your dataset with similar directory structure.

2.Modify the training script (train_model.py) to suit your needs.

3.Run the script and replace the old imageclassifier.h5 with the new model.

## 📡 AWS S3 and Twilio Setup
### AWS S3
• Make sure your S3 bucket is correctly configured with the necessary 
permissions to upload images.

• Update the bucket name in the .env file.
## Twilio
• Sign up for a Twilio account if you don't have one.

• Create a new Messaging Service and note down the SID and Auth Token.

• Add your Twilio phone number and recipient phone number in the .env file.
## 📊 Monitoring and Logging
• Logs for detected violations and SMS notifications are stored locally.

• For enhanced monitoring, consider integrating with cloud-based logging solutions.


## 📝 License 

📝 License
This project is licensed under the MIT License - see the [LICENSE]() file for details.

## 👥 Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any features, improvements, or bug fixes.

## 📧 Contact
For any questions or inquiries, feel free to reach out:

• Email: yashwanthms2005@gmail.com

• LinkedIn: https://www.linkedin.com/in/yashwanth-m-s-b4a44327a

• GitHub: https://github.com/yashwanthms25