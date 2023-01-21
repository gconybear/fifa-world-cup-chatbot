import pickle
import streamlit as st 
import pandas as pd 
import numpy as np  

from model import Bot 

@st.cache(allow_output_mutation=True)
def grab_data(): 
    print('grabbing data...')
    return {"data": pd.read_csv('data/world_cup_data.csv'), 
    "embeddings": pickle.load(open('data/world_cup_embeddings.pkl', 'rb'))}

D = grab_data() 
df, embeddings = D['data'], D['embeddings']
    

def blank(): return st.text('')

st.subheader("FIFA World Cup Chatbot ⚽️") 
blank() 

#with st.form(key='form'): 

query = st.text_area("Ask me anything world cup related...") 
blank()

search = st.button("search") 

if search: 
    
    with st.spinner("Generating Response 🤖"): 
        ai = Bot(data=df, embeddings=embeddings) 
        res = ai.generate_response(query=query)
    
    blank() 
    st.markdown(res)
    #st.code(res, language='markdown')    

