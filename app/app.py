import streamlit as st
import pandas as pd
import numpy as np 
import joblib
import altair as alt
pipe_lr = joblib.load(open("model/emotion_analyer.pkl","rb"))
def predict_emotions(docx):
    result = pipe_lr.predict([docx])
    return result[0]

def get_prediction_prob(docx):
    result =pipe_lr.predict_proba([docx])
    return result
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main():
    st.title("Emotion Analyser App")
    menu = ['Home','Monitor','About']
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home Emotion In Text")
        with st.form(key ="emotion_analyser_form"):
            raw_text = st.text_area("Type Here..")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2 = st.columns(2)
            Prediction = predict_emotions(raw_text)
            Probability = get_prediction_prob(raw_text)
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[Prediction]
                st.write("{}:{}".format(Prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(Probability)))
            with col2:
                st.success("Prediction Probability")   
                st.write(Probability)
                proba_df = pd.DataFrame(Probability,columns=pipe_lr.classes_)  
                st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]   

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability')
                st.altair_chart(fig,use_container_width=True)
    elif choice == "Monitor":
        st.subheader("Monitor App")
    else:
        st.subheader("About")    



if __name__ == "__main__":
    main()