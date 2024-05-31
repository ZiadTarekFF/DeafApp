import streamlit as st
from PIL import Image
import os 
import tensorflow as tf
import imageio
import urllib.request

from utils import load_data, num_to_char
from modelutil import load_model

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Select Model"

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Main page
if st.session_state.page == "Select Model" or st.session_state.page == "Back":
    st.title('Select Model')
    st.image("https://static.vecteezy.com/system/resources/previews/006/569/510/non_2x/i-love-you-hand-sign-with-heart-design-creative-valentine-day-card-with-inclusive-hand-heart-and-branches-element-design-vector.jpg", use_column_width=True)
    st.write("Please select a model:")

    if st.button("Lip Read"):
        st.session_state.page = "Lip Read"

    if st.button("Sign Language"):
        st.session_state.page = "Sign Language"

    if st.button("Speech to Text"):
        st.session_state.page = "Speech to Text"

# Lip Read Model
if st.session_state.page == "Lip Read":
    st.title('Lip Read')
    st.write("Lip Read model selected.")

    options = os.listdir(os.path.join('..', 'data', 's1'))
    selected_video = st.selectbox('Choose video', options)

    # Generate two columns 
    col1, col2 = st.columns(2)

    if options: 
        # Rendering the video 
        with col1: 
            st.info('The video below displays the converted video in mp4 format')
            file_path = os.path.join('..','data','s1', selected_video)
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

            # Rendering inside of the app
            video = open('test_video.mp4', 'rb') 
            video_bytes = video.read() 
            st.video(video_bytes)

        with col2: 
            st.info('This is all the machine learning model sees when making a prediction')
            video, annotations = load_data(tf.convert_to_tensor(file_path))
            imageio.mimsave('animation.gif', video, fps=10)
            st.image('animation.gif', width=400) 

            st.info('This is the output of the machine learning model as tokens')
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)

            # Convert prediction to text
            st.info('Decode the raw tokens into words')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)

    st.write("")
    if st.button("Back"):
        st.session_state.page = "Select Model"

# Sign Language Model
if st.session_state.page == "Sign Language":
    st.title('Sign Language')

    # Placeholder for Sign Language model
    st.write("Sign Language model selected.")
    st.image("https://learn2sign.vercel.app/about.gif", use_column_width=True)
    st.write("Upload a video of sign language:")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])

    if uploaded_file is not None:
        # You can add code here to process the uploaded video
        st.write("Video uploaded successfully!")

    st.write("")
    if st.button("Back"):
        st.session_state.page = "Select Model"

# Speech to Text Model
if st.session_state.page == "Speech to Text":
    st.title('Speech to Text')

    # Placeholder for Speech to Text model
    st.image('https://cdn.dribbble.com/users/1144754/screenshots/2978120/is-to-identify.gif', use_column_width=True)
    st.write("Speech to Text model selected.")

    st.write("Upload or record an audio:")
    uploaded_audio_file = st.file_uploader("Upload an audio file...", type=["mp3", "wav",'mp4','mpg'])
    if uploaded_audio_file is not None:
        # You can add code here to process the uploaded audio file
        st.write("Speech uploaded successfully!")

    # Define a reactive button to toggle recording state
    record_state = st.button("Record Audio 🎤")
    if record_state:
        st.write("Recording audio...")  # Placeholder for recording functionality
    else:
        st.write("Click to start recording")  # Placeholder when not recording

    st.write("Transcription of the audio:")
    st.write("The transcription of the audio will be displayed here.")

    # Text-to-speech section
    tts_text = st.text_area("Enter text for Text-to-Speech:")
    lang = st.selectbox("Select language:", options=['en', 'fr', 'de'])  # Add more languages as needed
    if st.button("Convert to Speech"):
        if tts_text:
            audio_file = text_to_speech(tts_text, lang)
            st.audio(audio_file, format='audio/mp3')
            os.remove("output.mp3")  # Delete the generated file after playing
        else:
            st.warning("Please enter some text to convert to speech.")

    st.write("")
    if st.button("Back"):
        st.session_state.page = "Select Model"
