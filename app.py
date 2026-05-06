import streamlit as st
import pandas as pd
import plotly.express as px

from src.predictor import load_model, load_class_names, predict_banana

st.set_page_config(
    page_title= "BananaLife AI",
    page_icon= "🍌",
    layout= "wide"
)

st.title("🍌 Banana Life AI")
st.write("Deep Learning powered banana ripeness and shelf life prediction system")

@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_class_names():
    return load_class_names()

model= get_model()
class_names= get_class_names()

uploaded_file= st.file_uploader(
    "Upload a banana image",
    type= ["jpg", "jpeg", "png"]
)

if uploaded_file:
    result= predict_banana(uploaded_file, model, class_names)
    col1, col2= st.columns(2)
    
    with col1:
        st.subheader("Uploaded banana Image")
        st.image(uploaded_file, use_container_width=True)
    
    with col2:
        st.subheader(" AI prediction")
        
        st.metric("Ripeness Stage", result['stage'].upper())
        st.metric("Model Confidence", f"{result['confidence']}%")
        st.metric("Estimated Days Left", result["days_left"])
        st.metric("Freshness Score", f"{result['freshness_score']} /100")
        
        st.info(result["advice"])
        
    st.divider()
    st.subheader("Prediction probability Breakdown")
    
    pred_df= pd.DataFrame({
        "Class": list(result["all_predictions"].keys()),
        "probability": list(result["all_predictions"].values())
        
    })
    
    fig= px.bar(
        pred_df,
        x= "Class",
        y= "probability",
        text= "probability",
        title= "class-wise Model prediction confidence"
    )
    
    st.plotly_chart(fig, use_container_width= True)
    st.divider()
    st.subheader("shelf Life Mapping")
    
    mapping_df= pd.DataFrame({
        "stage": ["Unripe", "Ripe", "Overripe", "Rotten"],
        "Estimated Days Left": ['5-7 days', '2-4 days', '1-2 days', '0 days'],
        "Freshness Score": [95, 80,45, 10 ]
        
    })
    
    st.dataframe(mapping_df, use_container_width= True)
else:
    st.info("Upload a banana image to predict ripeness stage and days left.")