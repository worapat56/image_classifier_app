
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle
import streamlit as st

                  


#load model 
with open('model.pkl','rb') as f:
    model = pickle.load(f)
    
#application title
st.title('Image CLasscification with MobileNetV2 by Boonyawat Jitratthanasaweat 6531501074')

#file uploader
uploaded_file = st.file_uploader('Upload an image...',type = ['jpg','jpeg','png'])

if uploaded_file is not None:
    #dispaly image
    img = Image.open(uploaded_file)
    st.image(img,caption='Uploaded Image',use_column_width=True)
        
    # Preprocess the image
    img = img.resize((224, 224))
    x =image.img_to_array(img)
    x = np.expand_dims(x, axis = 0) 
    x = preprocess_input(x)
    
    #Prediction 
    preds =model.predict(x)
    top_preds = decode_predictions(preds, top= 3)[0]
    
    #display prediction 
    st.subheader('Prediction:')
    for i, pred in enumerate(top_preds):
        st.write(f'{i+1}.**{pred[1]}** - {round(pred[2]*100,2)}%')
    
    
    
    
