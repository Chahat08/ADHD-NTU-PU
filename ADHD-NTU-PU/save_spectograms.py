import eeg_tools as et
import os
from PIL import Image
import matplotlib.pyplot as plt


spec_base = os.path.normpath("D:\EEG_SPECTOGRAMS_AAR")

spec_eo_adhd = os.path.join(spec_base, 'ADHD_EO')
spec_ec_adhd = os.path.join(spec_base, 'ADHD_EC')
spec_eo_cont = os.path.join(spec_base, 'CONTROL_EO')
spec_ec_cont = os.path.join(spec_base, 'CONTROL_EC')


for f in os.listdir(et.aar_eo_adhd): 
        sid = f.split('_')[1].split('.')[0]
        
        for i in range(1,21):
            data = et.get_single_channel_data(f'EO_A_{sid}_aar', channel_no=i)
        
            for j in range(1,10):
                # saving spectograms of 20 second segments for each subject
                plt.figure(figsize=(4,4))
                plt.specgram(x=data[j-1:j*20*256], Fs=256, cmap='jet', NFFT=128, noverlap=64)
                plt.axis('off')
                # file name = sid_channel_segment
                plt.savefig(spec_eo_adhd+f'/{sid}_c{i}_s{j}.png', bbox_inches='tight',transparent=True, pad_inches=0)
        
for f in os.listdir(et.aar_ec_adhd): 
        sid = f.split('_')[1].split('.')[0]
        
        for i in range(1,21):
            data = et.get_single_channel_data(f'Ec_A_{sid}_aar', channel_no=i)
        
            for j in range(1,10):
                # saving spectograms of 20 second segments for each subject
                plt.figure(figsize=(4,4))
                plt.specgram(x=data[j-1:j*20*256], Fs=256, cmap='jet', NFFT=128, noverlap=64)
                plt.axis('off')
                # file name = sid_channel_segment
                plt.savefig(spec_ec_adhd+f'/{sid}_c{i}_s{j}.png', bbox_inches='tight',transparent=True, pad_inches=0)
        
for f in os.listdir(et.aar_ec_cont): 
        sid = f.split('_')[1].split('.')[0]
        
        for i in range(1,21):
            data = et.get_single_channel_data(f'Ec_c_{sid}_aar', channel_no=i)
        
            for j in range(1,10):
                # saving spectograms of 20 second segments for each subject
                plt.figure(figsize=(4,4))
                plt.specgram(x=data[j-1:j*20*256], Fs=256, cmap='jet', NFFT=128, noverlap=64)
                plt.axis('off')
                # file name = sid_channel_segment
                plt.savefig(spec_ec_cont+f'/{sid}_c{i}_s{j}.png', bbox_inches='tight',transparent=True, pad_inches=0) 
        
for f in os.listdir(et.aar_eo_cont): 
        sid = f.split('_')[1].split('.')[0]
        
        for i in range(1,21):
            data = et.get_single_channel_data(f'Eo_c_{sid}_aar', channel_no=i)
        
            for j in range(1,10):
                # saving spectograms of 20 second segments for each subject
                plt.figure(figsize=(4,4))
                plt.specgram(x=data[j-1:j*20*256], Fs=256, cmap='jet', NFFT=128, noverlap=64)
                plt.axis('off')
                # file name = sid_channel_segment
                plt.savefig(spec_eo_cont+f'/{sid}_c{i}_s{j}.png', bbox_inches='tight',transparent=True, pad_inches=0)         