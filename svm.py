
# In[7]:

audio_data=[]
labels=[]
saprt=[]
path1=[]
import pickle
import pandas as pd
import librosa, os, glob    
import numpy as np
import os
import glob
from io import BytesIO
import matplotlib.pyplot as plt
zip_file='zip_file.zip'
from zipfile import ZipFile
import zipfile
with ZipFile(zip_file, 'r') as f:
      names = f.namelist()
audio_file=None
#%%
for fn in names:
    if not os.path.basename(fn):
        continue
    
    if fn.startswith('clf/d('):
        labels.append(0)
        audio_file,sr=librosa.load(fn)
    elif fn.startswith('clf/n('): 
        labels.append(1)
        audio_file,sr=librosa.load(fn)
    elif fn.startswith('clf/nide('):
        labels.append(2)
        audio_file,sr=librosa.load(fn)
    elif fn.startswith('clf/nedi('):
        labels.append(3)
        audio_file,sr=librosa.load(fn)
    else:
        print('unknown')   
    if audio_file is not None:
        audio_data.append(audio_file)
        saprt.append(sr)
        
        

# In[20]:plotting


fig = plt.figure(figsize=(14,6))
plt.plot(audio_data[2])    

fig = plt.figure(figsize=(14,6))
plt.plot(audio_data[1]) 

fig = plt.figure(figsize=(14,6))
plt.plot(audio_data[3]) 
fig = plt.figure(figsize=(14,6))
plt.plot(audio_data[4]) 
# In[28]:
import pywt
from sklearn.decomposition import PCA
pca=PCA(n_components=1)
features =np.empty((0,160))
scales=np.arange(1,161)
for ind in range(len(audio_data)):
    print('.',end='')
    coeff,freqs = pywt.cwt(audio_data[ind],scales,'morl')
    features =np.vstack([features,pca.fit_transform(coeff).flatten()])
#%%
coeff1,freqs = pywt.cwt(audio_data[1],scales,'morl',1/22050)
plt.matshow(coeff1)
plt.show()
coeff2,freqs = pywt.cwt(audio_data[2][:25000],scales,'morl')
coeff3,freqs = pywt.cwt(audio_data[3][:25000],scales,'morl')
coeff4,freqs = pywt.cwt(audio_data[4][:25000],scales,'morl')
#plt.imshow(coeff, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',vmax=abs(coeff).max(), vmin=-abs(coeff).max())  # doctest: +SKIP
#plt.show() # doctest: +SKIP

#%%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig=plt.figure(figsize=(40,15))
ax1= fig.add_subplot(1,2,1,projection='3d')
Y=np.arange(0,160,1)
X=np.arange(1,25001,1)
X, Y=np.meshgrid(X,Y)
ax1.plot_surface(X,Y,coeff2,cmap=cm.coolwarm,linewidth=0,antialiased=True)
ax1.set_xlabel("Time",fontsize=20)
ax1.set_ylabel("Scale",fontsize=20)
ax1.set_zlabel("Amplitude",fontsize=20)

plt.show()
# In[30]:
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn import svm  
features=features.transpose()
X_train,X_test,y_train,y_test =train_test_split(features,labels,test_size=0.2,random_state=4939)
#print(labelst)
#clf = svm.SVC(probability=True) 
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=4,random_state=0)

clf.fit(X_train,y_train)
#%%
X_cross=X_train[0:10]
Y_cross=y_train[0:10]
y_pred = clf.predict(X_cross)
#y_pred1 = clf.predict(X_test)
y_pred2= clf.predict(X_train)
a=accuracy_score(y_train,y_pred2)
a1=accuracy_score(Y_cross,y_pred)
print('acc='+str(a*100),'valacc='+str(a1*100))
#%%
 
print(confusion_matrix(y_train,y_pred2))
print(classification_report(Y_cross,y_pred))

pickle.dump(clf,open('model1.pkl','wb'))


cm1 = confusion_matrix(y_train,y_pred2)
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy


sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)
# In[47]:

def predict():
    audio_datat=[]
    labelst=[]
    saprtt=[]
    path1t=[]
    audio_filet=None
    labelst.append(0)
    audio_filet,srt=librosa.load('p1/d(1).wav')
    audio_datat.append(audio_filet)
    saprtt.append(srt)
    labelst.append(1)
    audio_filet,srt=librosa.load('p1/n(1).wav')
    audio_datat.append(audio_filet)
    saprtt.append(srt)
    labelst.append(2)
    audio_filet,srt=librosa.load('p1/nide(1).wav')
    audio_datat.append(audio_filet)
    saprtt.append(srt)
    labelst.append(3)
    audio_filet,srt=librosa.load('p1/nedi(1).wav')
    audio_datat.append(audio_filet)
    saprtt.append(srt)
    featurest =np.empty((0,160))
    
    scalest=np.arange(1,161)
    for ind in range(len(audio_datat)):
        print('.',end='')
        coefft,freqst = pywt.cwt(audio_datat[ind],scales,'morl')
        featurest =np.vstack([featurest,pca.fit_transform(coefft).flatten()])
    X_testt=featurest
    y_pred = clf.predict(X_testt)
    y_pred1= clf.predict_proba(X_testt)
    #df=pd.DataFrame(featurest,labelst)
    df=pd.DataFrame(y_pred1,labelst)
    
    tv= df[0][0]*500
    rv=df[1][1]*1200
    erv=df[2][2]*1200
    irv=df[3][3]*3300
    
    df=df.append({'tv':"",'rv':"",'irv':"",'erv':"",'IC':"",'TLC':"",'VC':"",'FRC':""}, ignore_index=True)
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
    
    if (rv>=700)and(rv<=1200):
        d="normal"
    else:
        d="ab_normal"
    #df2 = {'tv':a, 'irv': b, 'erv': c, 'rv':d}
    #df = df.append(df2, ignore_index=True)
    
    
    df['tv'][3]=a
    df['irv'][3]=b
    df['erv'][3]=c
    df['rv'][3]=d
    df=df.fillna('')
    print(df)
    df.to_csv('predicted_name.csv', index=False)

#%%
predict()
#%%
from scipy.signal import butter, filtfilt
import numpy as np

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

rawdata = audio_data[0]
signal = rawdata
fs = 22050

cutoff = 50

order = 7
conditioned_signal = butter_highpass_filter(signal, cutoff, fs, order)
fig = plt.figure(figsize=(14,6))
plt.plot(audio_data[0]) 
fig = plt.figure(figsize=(14,6))
plt.plot(conditioned_signal) 
#%%
coeff1,freqs = pywt.cwt(audio_data[2],scales,'morl')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig=plt.figure(figsize=(40,15))
ax1= fig.add_subplot(1,2,1,projection='3d')
Y=np.arange(0,160,1)
X=np.arange(1,len(audio_data[2])+1,1)
X, Y=np.meshgrid(X,Y)
ax1.plot_surface(X,Y,coeff1,cmap=cm.coolwarm,linewidth=0,antialiased=True)
ax1.set_xlabel("Time",fontsize=20)
ax1.set_ylabel("Scale",fontsize=20)
ax1.set_zlabel("Amplitude",fontsize=20)
