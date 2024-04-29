import streamlit as st
import tempfile
import numpy as np
from matplotlib import pyplot as plt
import os
# os.chdir("..")
from modules.sem import SEM
import pandas as pd


LIST_OF_IMAGE_EXT = [".jpeg",".tiff",".png",".tif",".jpg"]
DISABELTED = False

st.set_page_config(page_title="SEM feature detection", page_icon="📈")
st.markdown("# SEM feature detection")

@st.cache_data()
def load_data(): 
    try:
        sem = SEM()
        return sem
    except Exception as e:
        st.write(f"Error: {e}")
        return None

    

sem = load_data()


def display_result(im_path):
    if not sem:
        return 
    fig, patterns,class_name = sem.prediction(im_path)
    st.write(f"pattern detected : {class_name}")
    st.pyplot(fig) 
    df = pd.DataFrame(patterns,columns = ["Bottom CD","Mid CD","To CD"])
    st.dataframe(df,hide_index=True)

def run_random_image():
    if not sem:
        return 

    with st.spinner('In progress...'):
        data_images =  [x for x in os.listdir(sem.paths['image_dir']) if x.endswith('.png') 
                    or x.endswith('.jpg') ]
        idx=np.random.randint(0,len(data_images)) 
        im_path = os.path.join(sem.paths['image_dir'],data_images[idx])
        display_result(im_path)

    
   
def run_mulitple_file(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    with st.spinner('In progress...'):
        for uploaded_file in uploaded_files:
            im_path = os.path.join(temp_dir, uploaded_file.name)
            with open(im_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            if not os.path.isfile(im_path):
                assert False
            if os.path.splitext(im_path)[1].lower() not in LIST_OF_IMAGE_EXT:
                continue
            display_result(im_path)


st.button("Run a random image", 
          on_click=run_random_image)

uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
run_mulitple_file(uploaded_files)

   


