"""
# Non-Commercial Use Only
# This code is intended solely for academic research, personal study, or non-profit use. 
# Any commercial use, including but not limited to sales, profit-driven services, or product development, is strictly prohibited.
# Unauthorized commercial use of this code may result in legal action by the author.
# For any commercial use, please contact the author to obtain written permission.
"""

# Import necessary libraries
import torch
import numpy as np
import cv2  # OpenCV for image processing
from facenet_pytorch import MTCNN  # MTCNN for face detection and landmarks

# Initialize the MTCNN model for face detection and landmarks
mtcnn = MTCNN(keep_all=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Function to detect faces, landmarks, and align faces
def process_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    print("Image loaded successfully.")

    # Convert the image from BGR (OpenCV format) to RGB (MTCNN format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Converted image to RGB format.")

    # Detect faces and landmarks in the image
    faces, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)
    print("Faces and landmarks detected.")

    # Check if any faces were detected
    if faces is not None:
        # Iterate through detected faces
        for i, (box, prob, landmark) in enumerate(zip(faces, probs, landmarks)):
            print(f"Processing face {i+1}:")
            print(f"Bounding box: {box}")
            print(f"Landmarks: {landmark}")

            # Draw the bounding box on the original image
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.imwrite(f'output_step1_face_{i+1}_bbox.jpg', img)
            print(f"Step 1: Bounding box drawn and saved as 'output_step1_face_{i+1}_bbox.jpg'.")

            # Draw each landmark point (left eye, right eye, nose, left mouth, right mouth)
            for point in landmark:
                cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
            cv2.imwrite(f'output_step2_face_{i+1}_landmarks.jpg', img)
            print(f"Step 2: Landmarks drawn and saved as 'output_step2_face_{i+1}_landmarks.jpg'.")

            # Extract the face from the image using the bounding box
            face = img_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            cv2.imwrite(f'output_step3_face_{i+1}_extracted.jpg', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            print(f"Step 3: Extracted face saved as 'output_step3_face_{i+1}_extracted.jpg'.")

            # Perform face alignment (aligning based on eyes)
            left_eye = landmark[0]
            right_eye = landmark[1]

            # Calculate the angle between the eyes
            eye_delta_x = right_eye[0] - left_eye[0]
            eye_delta_y = right_eye[1] - left_eye[1]
            angle = np.arctan2(eye_delta_y, eye_delta_x) * 180 / np.pi
            print(f"Angle between eyes: {angle:.2f} degrees.")

            # Get the center between the eyes
            eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

            # Get the rotation matrix for the affine transformation
            rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
            print("Rotation matrix calculated for face alignment.")

            # Apply the affine transformation to align the face
            aligned_face = cv2.warpAffine(img_rgb, rotation_matrix, (img_rgb.shape[1], img_rgb.shape[0]))

            # Crop the aligned face using the bounding box
            aligned_face = aligned_face[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            print("Face aligned and cropped successfully.")

            # Convert aligned face back to BGR for display and save
            aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'output_step4_face_{i+1}_aligned.jpg', aligned_face_bgr)
            print(f"Step 4: Aligned face saved as 'output_step4_face_{i+1}_aligned.jpg'.")

    # Convert the image back to BGR for OpenCV display and return
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr

# Path to the input image
image_path = 'input_image.jpg'  # Replace with your image path

# Process the image to detect faces, landmarks, and align faces
processed_image = process_image(image_path)

# Display the processed image using OpenCV
cv2.imshow('Processed Image', processed_image)

# Wait for a key press and then close the displayed image window
cv2.waitKey(0)
cv2.destroyAllWindows()
