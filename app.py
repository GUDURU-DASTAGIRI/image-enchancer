import streamlit as st
import cv2
import numpy as np
from gfpgan import GFPGANer

# Initialize GFPGAN model
model = GFPGANer(model_path='/Users/nani/Downloads/GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None)

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

            # Display original and restored images
            st.image([input_img, restored_img], caption=["Original Image", "Restored Image"], channels="BGR", width=300)

if __name__ == "__main__":
    main()
