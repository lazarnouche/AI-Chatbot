import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
from modules.etch import Etch



st.set_page_config(page_title="Etch Profiles", page_icon="🐾")
st.markdown("# Etch Profiles")

etch = Etch()
ids = etch.df_rec[etch.df_rec.Path != ''].layer_ID.unique()
df = etch.df_lay[etch.df_lay.layer_ID.isin(ids)]
selected_query = st.selectbox("Select an etch question",df.Query)
columns_to_drop =  etch.unities.columns[etch.unities.astype('str').eq("-1").all()]

for txt in df.Profile[df.Query == selected_query].astype(str):
    st.write(txt)
    
# @st.cache_data(persist=True)
def pre_run_etch(query):  
    try:
        etch.handle_actions(query)

        return etch.variables
    except (FileNotFoundError, ValueError,IndexError) as e:
        print(f"Error: {e}")
        return []

def run_etch(var):
    try:

        return etch.diplay_mask(var)
    
    except (FileNotFoundError, ValueError,IndexError) as e:
        print(f"Error: {e}")
        return {}, f"Error: {e}"
    

vars = pre_run_etch(selected_query)


### ccheck when vars is not empty
option = st.selectbox(
   "Variables",
   etch.variables,
   index=0,
   placeholder="",
)

# st.write('You selected:', option)

figs_dic, text = run_etch(option)
var = text.split(":")[0]
keys = [str(x) for x in list(figs_dic.keys())]


# edited_df = st.experimental_data_editor(etch.df_lay.Query)

def run_widgets():

    with st.expander('',expanded=True):

        plot = st.container()

        if figs_dic == {}:
            st.write(text)
            return

        
        # use st.slider to select
        selected_option = st.select_slider(' ', 
                                        keys,
                                        value =keys[0])

        # selected_option = keys[selected_index]

        st.write(text.format(selected_option))

        # use st.pyplot
        with plot:
            st.pyplot(figs_dic[selected_option])
        
        
        st.dataframe(etch.dic_results[var].drop(columns=columns_to_drop),hide_index=True)

run_widgets()