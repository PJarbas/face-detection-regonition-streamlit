# importing libraries
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

#!pip install retina-face
#!pip install deepface

from retinaface import RetinaFace
from deepface import DeepFace


def detect(image):

    faces = RetinaFace.detect_faces(image)
    
    for _, face in faces.items():
        facial_area = face["facial_area"]
        # landmarks = face["landmarks"]
    
        #highlight facial area
        cv2.rectangle(image, (facial_area[2], facial_area[3])
                      , (facial_area[0], facial_area[1]), (255, 0, 0), 2)
        
    # #highlight the landmarks
    # for k,v in landmarks.items():
    #     print(k)
    #     print(v, '\n')
    #     cv2.circle(img, tuple(map(int, v)), 1, (0, 255, 0), cv2.FILLED)
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), faces


def visualize_images(img1, img2):
    "Preserves the scale of the image"
    
    imgs = [img1, img2]
    imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(imgs[0])
    ax1.axis('off')
    ax2.imshow(imgs[1])
    ax2.axis('off')
    st.pyplot(fig)


# Here is the function for UI
def main():
    st.title("Face Detection & Recognition App")
    st.markdown('Build with Streamlit and Retinaface')
    
    choice = st.sidebar.radio("Select a Method:", ("Face Detection", "Face Recognition"))

    if choice == "Face Detection":
        
        st.subheader("Face Detection")
        
        image_file = st.file_uploader(
            "Upload image", type=['jpeg', 'png', 'jpg'])
        
        if image_file:
            
            image = cv2.imread(image_file.name)

            if st.button("Process"):
                result_img, result_faces = detect(image=image)
                st.image(result_img, use_column_width=True)
                st.success("Found {} faces\n".format(len(result_faces)))

    if choice == "Face Recognition":
        
        st.subheader("Face Recognition")
        
        image_file1 = st.file_uploader(
            "Upload the first image", type=['jpeg', 'png', 'jpg'])
        
        image_file2 = st.file_uploader(
            "Upload the second image", type=['jpeg', 'png', 'jpg'])
        
        
        if image_file1 and image_file2:
            
            image1 = cv2.imread(image_file1.name)
            image2 = cv2.imread(image_file2.name)
            
            if st.button("Process"):
                
                result = DeepFace.verify(image1, image2, detector_backend = 'retinaface')
                
                visualize_images(image1, image2)
                
                st.json(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
