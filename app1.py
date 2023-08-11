import streamlit as st
import cv2
import numpy as np
import os
import subprocess
from gfpgan import GFPGANer
import matplotlib.pyplot as plt

# Initialize GFPGAN model
model = GFPGANer(model_path='./GFPGANv1.3.pth', upscale=6, arch='clean', channel_multiplier=2, bg_upsampler=None)

# Function to read an image using OpenCV and convert it to RGB format
def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Function to display faces side by side
def display_faces(original_img, enhanced_img):
    # Assuming the enhanced_img contains faces, extract and display them
    # Modify this part based on your face extraction logic
    original_faces = detect_faces_and_crop(original_img)
    enhanced_faces = detect_faces_and_crop(enhanced_img)

    plt.figure(figsize=(25, 10))
    ax1 = plt.subplot(1, 2, 1)
    plt.title('Original Faces', fontsize=16)
    plt.axis('off')
    for face in original_faces:
        ax1.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    
    ax2 = plt.subplot(1, 2, 2)
    plt.title('Enhanced Faces', fontsize=16)
    plt.axis('off')
    for face in enhanced_faces:
        ax2.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    
    st.pyplot(plt)

# Function to detect faces and crop
def detect_faces_and_crop(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    cropped_faces = []
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        cropped_faces.append(cropped_face)
    
    return cropped_faces

# Streamlit app
def main():
    st.title("Image Upscaler with GFPGAN")

    # Upload input image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Read uploaded image
        input_img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Upscaling options
        upscale_factor = st.slider("Upscale Factor", min_value=2, max_value=4, value=2)

        # Process image
        if st.button("Upscale"):
            restored_img = model.enhance(input_img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5)[2]

            # Save the input_img and restored_img to temporary files
            temp_input_path = 'temp_input.jpg'
            temp_restored_path = 'temp_restored.jpg'
            cv2.imwrite(temp_input_path, input_img)
            cv2.imwrite(temp_restored_path, restored_img)

            # Call the enhance_image function to perform GFPGAN enhancement
            enhance_image(temp_input_path, temp_restored_path)

            # Load the enhanced image
            enhanced_img = cv2.imread(temp_restored_path)

            # Display original and enhanced images
            st.image([input_img, enhanced_img], caption=["Original Image", "Restored Image"], channels="BGR", width=300)

            # Display faces using your visualization function
            display_faces(input_img, enhanced_img)

            # Clean up temporary files
            os.remove(temp_input_path)
            os.remove(temp_restored_path)

# Function to call the enhance_image function
def enhance_image(input_image_path, output_image_path):
    command = [
        "/opt/homebrew/bin/python3",  # Modify the path to your Python interpreter
        "/Users/nani/Downloads/GFPGAN-1.3.8/inference_gfpgan.py",
        "-i", input_image_path,
        "-o", output_image_path,
        "-v", "1.3",
        "-s", "2",
        "--bg_upsampler", "realesrgan"
    ]
    subprocess.run(command)

if __name__ == "__main__":
    main()
