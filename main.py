import streamlit as st
from PIL import Image
from model import model

def uploader():
    uploaded_file = st.file_uploader(
                    label='Upload the image', 
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=False,
                    key='image-uploader'
                )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image",use_column_width=True)
        results =model.predict(image)
        # print(results)
        out = results['final']
        st.markdown(f"#### Class: {out}")

def app():
    st.title('Brain Tumor Classifier')
    uploader()


if __name__ == '__main__':
    app()