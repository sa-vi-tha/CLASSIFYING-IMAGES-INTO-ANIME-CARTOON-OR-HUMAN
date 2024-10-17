

import streamlit as st
import base64
from io import BytesIO
import pickle
from PIL import Image
import webbrowser

# Function to convert an image to a base64 string
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Load the three images you want to display on the side
side_image1 = Image.open("All_anime.jpg")  # Replace with your image path
side_image2 = Image.open("All_cartoon.jpg")  # Replace with your image path
side_image3 = Image.open("All_human.jpg")  # Replace with your image path

# Convert the images to base64
img_base64_1 = image_to_base64(side_image1)
img_base64_2 = image_to_base64(side_image2)
img_base64_3 = image_to_base64(side_image3)

# Set the custom CSS for placing the images on the left side of the page
st.markdown(
    f"""
    <style>
    .side-image-container {{
        position: fixed;
        top: 10%;
        left: 0;
        width: 350px; /* Adjust the container width as needed */
        padding: 10px;
    }}
    .side-image {{
        margin-bottom: 25px; /* Spacing between the images */
        width: 350px; /* Increase this value to make the images larger */
    }}
    </style>
    <div class="side-image-container">
        <img src="data:image/png;base64,{img_base64_1}" class="side-image">
        <img src="data:image/png;base64,{img_base64_2}" class="side-image">
        <img src="data:image/png;base64,{img_base64_3}" class="side-image">
    </div>
    """,
    unsafe_allow_html=True
)







# Set the title of the app
st.title("CLASSIFYING IMAGES INTO ANIME,CARTOON OR HUMAN")



# Add a markdown element
st.markdown("### What Am I?")



# Create three columns
col1, col2, col3 = st.columns(3)

# Define the desired height for images (in pixels)
image_height = 250  # Adjust the height as needed

# Function to resize images
def resize_image(image_path, target_height):
      img = Image.open(image_path)
      aspect_ratio = img.width / img.height
      new_width = int(target_height * aspect_ratio)
      return img.resize((new_width, target_height))

# Add images to each column with the same height
with col1:
      img1 = resize_image("micky_mouse.jpg", image_height)  # Replace with your image path
      st.image(img1, caption="Cartoon", use_column_width=False)

with col2:
      img2 = resize_image("Girl.jpg", image_height)  # Replace with your image path
      st.image(img2, caption="Human", use_column_width=False)

with col3:
      img3 = resize_image("naruto.jpg", image_height)  # Replace with your image path
      st.image(img3, caption="Anime", use_column_width=False)

# Set a background color using CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #D5D6EA;  /* Set your desired background color */
        color: #333;  /* Set text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)



import numpy as np
from keras.models import load_model
MODEL_PATH = 'your_model.h5'
model = load_model('my_cnn_model.h5')
CLASSES = ['ANIME', 'CARTOON', 'HUMAN']
st.markdown("### Image Classifier")

st.write('**"Join Us on a Fun Adventure as We Classify and Explore the Colorful Universe of Anime, Cartoon, and Human Images!"**')

st.write ('**''Which group do I belong to?''**' "whether you're looking at Anime, a Cartoon character, or a Human face, revealing the true identity hidden within each image.")

# Create three columns
col1, col2, col3 = st.columns(3)

# Define the desired height for images (in pixels)
image_height = 225  # Adjust the height as needed

# Function to resize images
def resize_image(image_path, target_height):
      img = Image.open(image_path)
      aspect_ratio = img.width / img.height
      new_width = int(target_height * aspect_ratio)
      return img.resize((new_width, target_height))

# Add images to each column with the same height
with col1:
      img1 = resize_image("tom.jpg", image_height)  # Replace with your image path
      st.image(img1,caption="Cartoon", use_column_width=False)

with col2:
      img2 = resize_image("slaman.jpg", image_height)  # Replace with your image path
      st.image(img2,caption="Human", use_column_width=False)

with col3:
      img3 = resize_image("Dragan_ball.jpg",image_height)  # Replace with your image path
      st.image(img3,caption="Anime", use_column_width=False)

st.write("Is it a whimsical Cartoon, a dynamic Anime character, or a real-life Human? Let our intelligent image classifier unveil the mystery!")

st.title("Predict Who am I?")
uploaded_image = st.file_uploader("Upload a image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    def preprocess_image(image, target_size):
        image = image.resize(target_size)
        image = image.resize((150,150))
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    processed_image = preprocess_image(img, target_size=(150, 150))  # Adjust based on your model
    st.write("Making prediction...")
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    st.write(f"✅ Prediction: {CLASSES[predicted_class]}",icon="🎉")
    st.markdown(f"""
        <div style="background-color: #F67280; padding: 10px; border-radius: 5px; text-align: center;">
            <h2 style="color: #01F9C6;"> {CLASSES[predicted_class]}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    if st.button('feedback'):
        feedback = st.radio('Was the prediction correct?', ['Yes', 'No'])
        if feedback == 'Yes':
            st.write('Glad to hear that!')
        else:
            st.write('We will keep improving our model!')
        st.write("If you have any questions or comments,mail:comment@gmail.com, feel free to reach out to us.")
        st.write("Open Google Colab")
        # Display instructions
        st.write("Click the button below to open the Google Colab project.")
        # Add a button to open the Google Colab link
        if st.button('Open Google Colab'):
            colab_url = "https://colab.research.google.com/drive/1zCW46ZYUWIssQEeDDNbTkBNhqdjp5wFc"
            webbrowser.open_new_tab('colab_url')











