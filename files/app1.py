import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.applications import EfficientNetB7
from keras.models import Sequential
from PIL import Image
import pandas as pd

efficientnetB7= EfficientNetB7(include_top=False,weights='imagenet',input_shape=(150, 150, 3))

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),  
      tf.keras.metrics.AUC(name='auc')
]

model2 = Sequential()
model2 = efficientnetB7.output
model2 = tf.keras.layers.GlobalAveragePooling2D()(model2)
model2 = tf.keras.layers.Dropout(0.5)(model2)
model2 = tf.keras.layers.Dense(5, activation='softmax')(model2)
model2 = tf.keras.models.Model(inputs=efficientnetB7.input, outputs=model2)

model2.compile( optimizer='adam',
    loss='categorical_crossentropy',
    metrics=METRICS)

hist = model2.load_weights('C:\\Users\\Admin\\AppData\\Local\\Programs\\Python\\Python311\\My Programs\\Alzheimer Disease Detection Using Ensemble Learning\\efficientb7.h5')

# Define the labels
labels = ['Final AD JPEG','Final CN JPEG', 'Final EMCI JPEG', 'Final LMCI JPEG','Final MCI JPEG']

# Set page title and favicon
custom_theme = {
    "base": "light",
    "backgroundColor": "#020101",
    "textColor": "#0db9e6"
}

# Set the page config to use the custom theme
st.set_page_config(page_title="Alzheimer's Detection App", page_icon=":brain:")

# Set background color and page width
st.markdown(
    """
    <style>
       body {
            color: #0db9e6;
            background-color: #020101;
        }
        .css-1g6e0jr {
            color: #FFFFFF;
        }
        .stSidebar {
            background-color: #000000;
            border-radius: 10px;
            margin-top: 30px;
            padding: 20px;
            color: #0db9e6;
        }
        .stImage {
            border-radius: 10px;
            box-shadow: 0 0 10px #ccc;
        }
        .stSelectbox option {
        color: #0db9e6;
        font-weight: bold;
        }
        .stButton button {
            background-color: #008080; 
            color: #fff;
            border-radius: 20px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #009999;
            box-shadow: 0 0 10px #ccc;
        }
        [data-testid="stSidebar"][aria-expanded="true"] {
        background-color: #000000;
        color: #0db9e6;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    st.title('Alzheimer\'s Disease Detection using Ensemble Learning')
    st.write('Upload an MRI brain image to predict the likelihood of Alzheimer\'s disease.')


    # Create a file uploader
    uploaded_file = st.file_uploader('Choose an MRI image', type=['jpg', 'jpeg', 'png'])
    # If the user uploads a file
    if uploaded_file is not None:
        if uploaded_file.type in ['image/jpeg', 'image/jpg', 'image/png']:
            image = Image.open(uploaded_file)
            new_width = int(image.width * (300 / image.height))
            resized_image = image.resize((new_width, 200))
            st.success("Successfully Uploaded")
            st.image(resized_image, caption='Uploaded Image', use_column_width=True)
            # Preprocess the image
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image = cv2.resize(image,(150, 150))
            image = image.reshape(1, 150, 150, 3)
            prd = model2.predict(image)
            prd = np.argmax(prd, axis = 1)[0]


            if prd == 0:
                prd = "AD"
            elif prd == 1:
                prd = "CN"
            elif prd == 2:
                prd = "EMCI"
            elif prd == 3:
                prd = "LMCI"
            elif prd == 4:
                prd = "MCI"
            if prd!=1:
                #print(f'Model Predict That is  a {prd}')       
                # Display the prediction result
                #st.write('Prediction: ', prd)
                st.write(
    '<div style="text-align: center; font-weight: bold; font-size: 24px;">Prediction: {}</div>'.format(prd),
    unsafe_allow_html=True)

        else:
            st.error('Please upload a valid image file (JPEG, JPG or PNG).')


def about():
    menus = ["AD", "MCI", "EMCI", "LMCI", "CN"]
    st.sidebar.write('<h1 style="color: #0db9e6; font-weight: bold;">Select an Option</h1>', unsafe_allow_html=True)
    choices = st.sidebar.selectbox(" ",menus)

    if choices == "AD":
        import ad
        ad.app()
    elif choices == "MCI":
        import mci
        mci.app()
    elif choices == "EMCI":
        import emci
        emci.app()
    elif choices == "LMCI":
        import lmci
        lmci.app()
    elif choices == "CN":
        import cn
        cn.app()

def visualizations():
    st.title("Visualizations")
    menus1 = ["Densenet", "CNN", "Efficient Net B7", "VGG 19", "Ensemble"]
    st.sidebar.write('<h1 style="color: #0db9e6; font-weight: bold;">Select an Option</h1>', unsafe_allow_html=True)
    choices1 = st.sidebar.selectbox(" ", menus1)

    if choices1 == "Densenet":
        menus2 = ["Accuracy", "Precision", "Loss", "AUC Score", "Confusion Matrix"]
        st.sidebar.write('<h1 style="color: #0db9e6; font-weight: bold;">Select an Option</h1>', unsafe_allow_html=True)
        choices2 = st.sidebar.selectbox(" ", menus2)
        if choices2 == "Accuracy":
            image = Image.open('images\densenet_accuracy.png')
            st.image(image, caption='Accuracy', use_column_width=True)
        elif choices2 == "Precision":
            image = Image.open('images\densenet_precision.png')
            st.image(image, caption='Precision', use_column_width=True)
        elif choices2 == "Loss":
            image = Image.open('images\densenet_loss.png')
            st.image(image, caption='Loss', use_column_width=True)
        elif choices2 == "AUC Score":
            image = Image.open('images\densenet_auc.png')
            st.image(image, caption='AUC Score', use_column_width=True)
        elif choices2 == "Confusion Matrix":
            image = Image.open('images\densenet_confusion.png')
            st.image(image, caption='Confusion Matrix', use_column_width=True)

    elif choices1 == "CNN":
        menus3 = ["Accuracy", "Precision", "Loss", "AUC Score", "Confusion Matrix"]
        st.sidebar.write('<h1 style="color: #0db9e6; font-weight: bold;">Select an Option</h1>', unsafe_allow_html=True)
        choices3 = st.sidebar.selectbox(" ", menus3)
        if choices3 == "Accuracy":
            image = Image.open('images\cnn_accuracy.png')
            st.image(image, caption='Accuracy', use_column_width=True)
        elif choices3 == "Precision":
            image = Image.open('images\cnn_precision.png')
            st.image(image, caption='Precision', use_column_width=True)
        elif choices3 == "Loss":
            image = Image.open('images\cnn_loss.png')
            st.image(image, caption='Loss', use_column_width=True)
        elif choices3 == "AUC Score":
            image = Image.open('images\cnn_auc.png')
            st.image(image, caption='AUC Score', use_column_width=True)
        elif choices3 == "Confusion Matrix":
            image = Image.open('images\cnn_confusion.png')
            st.image(image, caption='Confusion Matrix', use_column_width=True)

    elif choices1 == "Efficient Net B7":
        menus4 = ["Accuracy", "Precision", "Loss", "AUC Score", "Confusion Matrix"]
        st.sidebar.write('<h1 style="color: #0db9e6; font-weight: bold;">Select an Option</h1>', unsafe_allow_html=True)
        choices4 = st.sidebar.selectbox(" ", menus4)
        if choices4 == "Accuracy":
            image = Image.open('images\netb7_accuracy.png')
            st.image(image, caption='Accuracy', use_column_width=True)
        elif choices4 == "Precision":
            image = Image.open('images\netb7_precision.png')
            st.image(image, caption='Precision', use_column_width=True)
        elif choices4 == "Loss":
            image = Image.open('images\netb7_loss.png')
            st.image(image, caption='Loss', use_column_width=True)
        elif choices4 == "AUC Score":
            image = Image.open('images\netb7_auc.png')
            st.image(image, caption='AUC Score', use_column_width=True)
        elif choices4 == "Confusion Matrix":
            image = Image.open('images\netb7_confusion.png')
            st.image(image, caption='Confusion Matrix', use_column_width=True)

    elif choices1 == "VGG 19":
        menus5 = ["Accuracy", "Precision", "Loss", "AUC Score", "Confusion Matrix"]
        st.sidebar.write('<h1 style="color: #0db9e6; font-weight: bold;">Select an Option</h1>', unsafe_allow_html=True)
        choices5 = st.sidebar.selectbox(" ", menus5)
        if choices5 == "Accuracy":
            image = Image.open('images\vgg19_accuracy.png')
            st.image(image, caption='Accuracy', use_column_width=True)
        elif choices5 == "Precision":
            image = Image.open('images\vgg19_precision.png')
            st.image(image, caption='Precision', use_column_width=True)
        elif choices5 == "Loss":
            image = Image.open('images\vgg19_loss.png')
            st.image(image, caption='Loss', use_column_width=True)
        elif choices5 == "AUC Score":
            image = Image.open('images\vgg19_auc.png')
            st.image(image, caption='AUC Score', use_column_width=True)
        elif choices5 == "Confusion Matrix":
            image = Image.open('images\vgg19_confusion.png')
            st.image(image, caption='Confusion Matrix', use_column_width=True)

    elif choices1 == "Ensemble":
        menus6 = ["Accuracy", "Precision", "Loss", "AUC Score", "Confusion Matrix"]
        st.sidebar.write('<h1 style="color: #0db9e6; font-weight: bold;">Select an Option</h1>', unsafe_allow_html=True)
        choices6 = st.sidebar.selectbox(" ", menus6)
        if choices6 == "Accuracy":
            image = Image.open('images\ensemble_accuracy2.png')
            st.image(image, caption='Accuracy', use_column_width=True)
        elif choices6 == "Precision":
            image = Image.open('images\ensemble_precision2.png')
            st.image(image, caption='Precision', use_column_width=True)
        elif choices6 == "Loss":
            image = Image.open('images\ensemble_loss2.png')
            st.image(image, caption='Loss', use_column_width=True)
        elif choices6 == "AUC Score":
            image = Image.open('images\ensemble_auc2.png')
            st.image(image, caption='AUC Score', use_column_width=True)
        elif choices6 == "Confusion Matrix":
            image = Image.open('images\ensemble_confusion.png')
            st.image(image, caption='Confusion Matrix', use_column_width=True)



# Add the menu options to the sidebar
menu = ["Home", "About", "Visualizations"]
st.sidebar.write('<h1 style="color: #0db9e6; font-weight: bold;">Select an Option</h1>', unsafe_allow_html=True)
choice = st.sidebar.selectbox(" ", menu)

# Show the appropriate page based on the user's menu choice
if choice == "Home":
    main()
elif choice == "About":
    about()
elif choice == "Visualizations":
    visualizations()

