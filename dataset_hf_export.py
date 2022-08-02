#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os

dataset = np.load('dataset_hf_radio.npy')
tags = pd.read_csv('dataset_hf_radio_tags.csv')

df = pd.DataFrame(tags)
df.columns = ['idx', 'mode', 'snr']

sample_df = pd.DataFrame(columns=['idx', 'mode', 'snr'])
df = df.query('snr==25')
df = df[df["mode"].str.contains("lsb|olivia8_250|olivia16_500|olivia16_1000|rtty45_170|rtty50_170|psk63|dominoex11")==False]

for g in df['mode'].unique():
    temp_df = df.query('mode==@g').sample(n=1000) ## 그룹별 데이터 추출 및 2개 비복원 추출
    sample_df = pd.concat([sample_df,temp_df]) ## 데이터 추가

random_sample_idx = sample_df['idx'].unique()
mpl.rcParams[ 'figure.figsize' ] = ( 1, 1 )
mpl.rcParams[ 'figure.dpi' ] = 72

for i in random_sample_idx:
    mode = sample_df['mode'][sample_df['idx']==i].values[0]
    
    fig,ax = plt.subplots() 
    ax.margins(0)
    plt.axis('off')
    ax.plot(np.real(dataset[i,1:1024]),'b', linewidth=0.5)
    ax.plot(np.imag(dataset[i,1:1024]),'r', linewidth=0.5)
    plt.xlim([1, 1024])
    plt.ylim([-2.5, 2.5])
    fig = plt.gcf()
    fig.tight_layout()
    ax.set_axis_off()
    params = dict(bottom=0, left=0, right=1, top=1)
    fig.subplots_adjust(**params)

    filePath = os.path.join(mode)
    if not os.path.exists(filePath):
        os.makedirs(filePath)
    fileName = os.path.join(filePath, str(mode)+'_'+str(i)+'.png')
    plt.savefig(fileName)


# %%

dataset_median = np.abs(np.median(dataset, axis=1))
dataset_avg = np.abs(np.average(dataset, axis=1))
dataset_max = np.abs(np.max(dataset, axis=1))
dataset_min = np.abs(np.min(dataset, axis=1))
dataset_maxfreq = np.zeros(dataset.shape[0], dtype=np.float32)
dataset_second_maxfreq = np.zeros(dataset.shape[0], dtype=np.float32)
dataset_maxfreq_power = np.zeros(dataset.shape[0], dtype=np.float32)
dataset_second_maxfreq_power = np.zeros(dataset.shape[0], dtype=np.float32)

for i in range(dataset.shape[0]):
    dataset_maxfreq[i] = np.argmax(np.abs(np.fft.fftshift(np.fft.fft(dataset[i,:]))))
    dataset_second_maxfreq[i] = np.abs(np.fft.fftshift(np.fft.fft(dataset[i,:]))).argsort()[1]
    dataset_maxfreq_power[i] = np.max(np.abs(np.fft.fftshift(np.fft.fft(dataset[i,:]))))
    dataset_second_maxfreq_power[i] = sorted(np.abs(np.fft.fftshift(np.fft.fft(dataset[i,:]))))[-2]

dataset_stat = np.vstack((dataset_avg, dataset_median, dataset_max, dataset_min, dataset_maxfreq, dataset_maxfreq_power, dataset_second_maxfreq, dataset_second_maxfreq_power))
df_stat = pd.DataFrame(dataset_stat.T, columns = ['avg', 'median', 'max', 'min', 'maxfreq', 'maxfreq_power', 'second_maxfreq', 'second_maxfreq_power'])

df_save = pd.concat([df, df_stat], axis=1, join='inner')
df_save = df_save.drop(columns='idx')
df_save = df_save.drop(columns='snr')

df_save.to_csv('dataset_hf_stats.csv', index=False, sep='\t', encoding='utf-8')

# %%
