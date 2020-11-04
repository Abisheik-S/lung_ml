import streamlit as st
from PIL import Image

import pickle
import numpy as np
import librosa
import  pywt
from sklearn.decomposition import PCA
import pandas as pd


clf=pickle.load(open('model1.pkl','rb'))




df1=[]
@st.cache(suppress_st_warning=True)
def predict(normal,deep,nide,nedi):
    audio_datat=[]
    labelst=[]
    saprtt=[]
    path1t=[]
    audio_filet=None
    labelst.append(0)
    audio_filet,srt=librosa.load(normal)
    audio_datat.append(audio_filet)
    saprtt.append(srt)
    labelst.append(1)
    audio_filet,srt=librosa.load(deep)
    audio_datat.append(audio_filet)
    saprtt.append(srt)
    labelst.append(2)
    audio_filet,srt=librosa.load(nide)
    audio_datat.append(audio_filet)
    saprtt.append(srt)
    labelst.append(3)
    audio_filet,srt=librosa.load(nedi)
    audio_datat.append(audio_filet)
    saprtt.append(srt)
    featurest =np.empty((0,160))
    pca=PCA(n_components=1)
    scalest=np.arange(1,161)
    for ind in range(len(audio_datat)):
        print('.',end='')
        coefft,freqst = pywt.cwt(audio_datat[ind],scalest,'morl')
        featurest =np.vstack([featurest,pca.fit_transform(coefft).flatten()])
    X_testt=featurest
    y_pred = clf.predict(X_testt)
    y_pred1= clf.predict_proba(X_testt)
    df=pd.DataFrame(y_pred1,labelst)
    
    tv= df[0][0]*500
    rv=df[1][1]*1200
    erv=df[2][2]*1200
    irv=df[3][3]*3300
    df1=pd.DataFrame()
    df=df.append({'tv':"",'rv':"",'irv':"",'erv':"",'IC':"",'TLC':"",'VC':"",'FRC':""}, ignore_index=True)
    df1=df.append({'irv':"",'erv':"",'IC':"",'TLC':"",'VC':"",'FRC':""}, ignore_index=True)
    df['tv'][0]=300
    df['tv'][1]=500
    df['tv'][2]=tv
    
    df['rv'][0]=0
    df['rv'][1]=1200
    df['rv'][2]=rv
    
    df['erv'][0]=700
    df['erv'][1]=1200
    df['erv'][2]=erv
    
    df['irv'][0]=1900
    df['irv'][1]=3300
    df['irv'][2]=irv
    
    df['IC'][0]=irv+tv
    df['TLC'][0]=tv+rv+irv+erv
    df['VC'][0]=tv+irv+erv
    df['FRC'][0]=rv+erv
    if ((tv>=300) and (tv<=500)):
        a="normal"
    else:
           a="ab_normal"
           
    if ((irv>=1900) and (irv<=3300)):
        b="normal" 
    else:
        b="ab_normal"
    
    if ((erv>=700) and (erv<=1200)):
        c="normal"
    else:
        c="ab_normal"
    
    if (rv>=800)and(rv<=1200):
        d="normal"
    else:
        d="ab_normal"
    df['tv'][3]=a
    df['irv'][3]=b
    df['erv'][3]=c
    df['rv'][3]=d
    df=df.fillna('')
    
    return df

def main():
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    local_css("style.css")
    
    st.set_option('deprecation.showfileUploaderEncoding', False)
    html_temp = """
    <div style="background-color:#084C46 ;padding:20px">
    <h1 style="color:white;text-align:center;"> Lung Parameters Detection</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    deep=st.file_uploader("Deep inhale Deep Exhale", type=["wav", "mp4", "mp3"])
    normal = st.file_uploader("Normal inhale Normal Exhale", type=["wav", "mp4", "mp3"])
    nide = st.file_uploader("Normal inhale Deep Exhale", type=["wav", "mp4", "mp3"])
    nedi= st.file_uploader("Deep inhale Normal Exhale", type=["wav", "mp4", "mp3"])

    if st.button("Predict"):
       try:
        output=predict(deep,normal,nide,nedi)
        st.dataframe(output) 
       except:
        error_temp = """
        <div style="background-color:#F14125 ;padding:20px">
        <h1 style="color:white;text-align:center;"> Invalid input</h2>
        </div>
        """
        st.markdown(error_temp, unsafe_allow_html=True)

if __name__=='__main__':
    main()
