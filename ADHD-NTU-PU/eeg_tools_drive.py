import os
from mat4py import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

"SETTING PATHS"

# raw data paths
raw_base = os.path.normpath("/content/drive/MyDrive/EEG_DATA")
eo_adhd = os.path.join(raw_base, 'Eyes-open', 'ADHD')
ec_adhd = os.path.join(raw_base, 'Eyes-closed', 'ADHD')
eo_cont = os.path.join(raw_base, 'Eyes-open', 'Control')
ec_cont = os.path.join(raw_base, 'Eyes-closed', 'Control')

# channel locations path
locs_path = os.path.normpath("/content/drive/MyDrive/EEG_DATA/Standard-10-20-Cap19.locs")

# processed data paths
proc_base = os.path.normpath("/content/drive/MyDrive/EEG_DATA")

# ADJUST
adj_base = os.path.join(proc_base, "Adjust_matfiles")
adj_eo_adhd = os.path.join(adj_base, 'ADHD_EO')
adj_ec_adhd = os.path.join(adj_base, 'ADHD_EC')
adj_eo_cont = os.path.join(adj_base, 'CONTROL_EO')
adj_ec_cont = os.path.join(adj_base, 'CONTROL_EC')

#AAR
aar_base = os.path.join(proc_base, "AAR_matfiles")
aar_eo_adhd = os.path.join(aar_base, 'ADHD_EO')
aar_ec_adhd = os.path.join(aar_base, 'ADHD_EC')
aar_eo_cont = os.path.join(aar_base, 'CONTROL_EO')
aar_ec_cont = os.path.join(aar_base, 'CONTROL_EC')

"READING CHANNEL LOCATIONS"

# storing channels names in a list
channels = []
locs = open(locs_path)
for i in locs.read().split('\t'):
    if '\n' in i:
        channels.append(i.split('\n')[0].strip())

"FUNCTION DEFINITIONS"

# help function
def toolkit_help():
    print('REQUIRED FILE NAME FORMAT: [ EC/EO ]_[ C/A ]_[ ID ]_[ RAW/ADJ/AAR ]\t', 'eg: EC_A_103_RAW, eo_c_144_adj, ec_a_194_aar')
    print('--'*50,end='\n'+'--'*50+'\n')
    
    print('\t\t---INFORMATION FUNCTIONS---\t\t')
    print()
    print('FUNC: toolkit_help()','ARGS: none', 'USE: Display available functions and their uses and the file format reqd.', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: channel_names()','ARGS: none', 'USE: prints the names and numbers of channels', sep='\n', end='\n'+'--'*50+'\n')
    
    print('--'*50,end='\n'+'--'*50+'\n')
    
    print('\t\t---FILE  DATA RETRIEVAL FUNCTIONS---\t\t')
    print()
    print('FUNC: get_data_as_list(file)','ARGS: file name in reqd. format', 'USE: returns the matfile data as a (256x180) x 19 list', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: get_data_as_np_array(file)','ARGS: file name in reqd. format', 'USE: returns the matfile data as a (256x180) x 19 numpy array', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: get_data_as_pd_dataframe(file)','ARGS: file name in reqd. format', 'USE: returns the matfile data as a (256x180) x 19 numpy array', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: get_single_channel_data(file, channel_name='', channel_no=0)','ARGS: file name in reqd. format, channel_name or channel_no', 'USE: returns the data for a single channel as an np array of size (1,(256x180))', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: get_channel_data(file, **chans)','ARGS: file name in reqd. format, **KWARGS: channel_names=[] (list of channel names), channel_nos=[] (list of channel nos)', 'USE: returns the matfile data only for the specified channels, as dict of numpy arrays', sep='\n')
    print('--'*50,end='\n'+'--'*50+'\n')
    
    print('\t\t---SPECTROGRAMS---\t\t')
    print()
    print('DEFAULT ARGUMENTS: fs=256, cm=''jet'', nfft=512, noverlap=92, show_colorbar=False (COLORBAR SUPPORT NOT IMPLEMENTED)', end='\n'+'--'*50+'\n')
    print('FUNC: spec_single_channel(file, channel_name='', channel_no=0)','ARGS: file name in reqd format, channel_name/channel_no, default args', 'USE: plot spectrogram for the specified channel w the default args. (or specify vals)', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: spec_for_channels(file, channel_names=[],channel_nos=[])','ARGS: file name in reqd format, channel_names: list of channel names, channel_nos: list of channel numbers (ints), default args.', 'USE: plot spectrogram for the specified channel(s) w the default args. (or specify vals)', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: specs_for_file(file)','ARGS: file name in reqd format, default args.', 'USE: plot spectrogram for all 19 channels in the file', sep='\n', end='\n'+'--'*50+'\n')
    
    print('--'*50,end='\n'+'--'*50+'\n')
    
    print('\t\t---VISUALISATION FUNCTIONS---\t\t')
    print()
    print('FUNC: plot_all_in_one(file)','ARGS: file name in reqd. format', 'USE: plot all channel data in one graph, every channel centered at x-axis', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: plot_eeg(file)','ARGS: file name in reqd. format', 'USE: plot all channel data in one graph, seperated: main eeg-like plot', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: plot_single_channel(file, num=0, name='')','ARGS: file name in reqd. format, reqd. channel name or number', 'USE: plot just a single channel. pass it\'s name or number as kwarg', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: plot_channel_segment(file, num=0, name='', x1=0, x2=256*180)','ARGS: file name in reqd. format, reqd. channel name or number, segment start and segment end in seconds', 'USE: plot a zoomed in segment for a single channel, pass channel name/number and the segment start and end in seconds.', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: plot_all_single_channels(file)','ARGS: file name in reqd. format', 'USE: plot all 19 channels in seperate graphs', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: plot_3d(file, cm=\'magma\')','ARGS: file name in reqd. format, optional color map', 'USE: 3d plot for all channels for the full 3 mins. optional arg for color map.', sep='\n', end='\n'+'--'*50+'\n')
    
    
# auxilliary functions

# finds the path to a matfile
def resolve_path(eyes, label, sid, stat):
    found = False
    
    if stat=='raw':
        if eyes=='ec':
            if label=='a':
                file_path = ec_adhd
            else: file_path = ec_cont
        else:
            if label=='a':
                file_path=eo_adhd
            else: file_path=eo_cont
            
    elif stat=='adj':
        if eyes=='ec':
            if label=='a':
                file_path = adj_ec_adhd
            else: file_path = adj_ec_cont
        else:
            if label=='a':
                file_path=adj_eo_adhd
            else: file_path=adj_eo_cont
                
    elif stat=='aar':
        if eyes=='ec':
            if label=='a':
                file_path = aar_ec_adhd
            else: file_path = aar_ec_cont
        else:
            if label=='a':
                file_path=aar_eo_adhd
            else: file_path=aar_eo_cont

    
    for f in os.listdir(file_path):
        if sid in f:
            file_path = os.path.join(file_path, f)
            found = True
            
    if found: return file_path
    else:
        print('FILE NOT FOUND')
        
# returns the loaded matfile
"""
def load_file(eyes, label, sid, stat):
    found = False
    
    if stat=='raw':
        if eyes=='ec':
            if label=='a':
                file_path = ec_adhd
            else: file_path = ec_cont
        else:
            if label=='a':
                file_path=eo_adhd
            else: file_path=eo_cont
            
    elif stat=='adj':
        if eyes=='ec':
            if label=='a':
                file_path = adj_ec_adhd
            else: file_path = adj_ec_cont
        else:
            if label=='a':
                file_path=adj_eo_adhd
            else: file_path=adj_eo_cont
                
    elif stat=='aar':
        if eyes=='ec':
            if label=='a':
                file_path = aar_ec_adhd
            else: file_path = aar_ec_cont
        else:
            if label=='a':
                file_path=aar_eo_adhd
            else: file_path=aar_eo_cont

    
    for f in os.listdir(file_path):
        if sid in f:
            file_path = os.path.join(file_path, f)
            found = True
            
    if found: return file_path
    else:
        print('FILE NOT FOUND')

# converts channel name to list index
def channel_name_to_num(name):
    for i in range(len(channels)):
        if channels[i].lower()==name.lower():
            n=i
    return n
"""

"VISUALISATION FUNCTIONS"

# plot all channel data in one graph, every channel centered at x-axis
def plot_all_in_one(file):
    eyes, label, sid, stat = file.lower().split('_')
    if stat=='raw':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    
    plt.figure(figsize=(16,8))
    plt.grid()
    plt.plot(file)
    plt.legend(channels, ncol=2, bbox_to_anchor=(1.0, 1.0))
    plt.title(sid+' '+eyes.upper()+' '+label.upper()+' '+stat.upper())
    plt.xlabel('Seconds')
    plt.ylabel('Sensor Value (uV)')
    
# plot all channel data in one graph, seperated: main eeg-like plot
def plot_eeg(file):
    eyes, label, sid, stat = file.lower().split('_')
    if stat=='raw':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    
    plt.figure(figsize=(20,10), dpi=80)
    x=0
    for data in np.transpose(file):
        #plt.plot(list(chain(range(-1000, 0),range(len(data)))), [x]*(len(data)+1000), color='black', alpha=0.5)
        plt.plot(range(len(data)), [data[i]+x for i in range(len(data))], linewidth=0.8)
        x+=60
    plt.yticks(np.linspace(0, 19*60, num=19, endpoint=False), channels, size='large')
    plt.xticks(np.linspace(0, 256*180, num=20, endpoint=True), size='large')
    plt.grid()
    plt.title(sid+' '+eyes.upper()+' '+label.upper()+' '+stat.upper(), size='large')
    plt.legend(channels, ncol=1, bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('Seconds', size='large')
    plt.ylabel('Channel', size='large')
    
# plot just a single channel. pass it's name or number as kwarg
def plot_single_channel(file, num=0, name=''):
    if num==0 and name=='':
        print('NO CHANNEL PASSED')
        return
    
    eyes, label, sid, stat = file.lower().split('_')
    if stat=='raw':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    
    if num==0:
        n=channel_name_to_num(name)
    else: n=num-1 # cuz channel idx 0 = 1st channel
    
    plt.figure(figsize=(16,4))
    plt.grid()
    plt.plot(np.transpose(file)[n])
    plt.legend([channels[n]])
    plt.title(sid+' '+eyes.upper()+' '+label.upper()+' '+stat.upper()+' '+ channels[n], size='large')
    plt.xlabel('Seconds')
    plt.ylabel('Sensor Value (uV)')
    
# plot a zoomed in segment for a single channel, pass channel name/number and the segment start and end in seconds.
def plot_channel_segment(file, num=0, name='', x1=0, x2=256*180):
    if num==0 and name=='':
        print('NO CHANNEL PASSED')
        return
    
    eyes, label, sid, stat = file.lower().split('_')
    if stat=='raw':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    
    if num==0:
        n=channel_name_to_num(name)
    else: n=num-1 # cuz channel idx 0 = 1st channel
    
    plt.figure(figsize=(16,4))
    plt.grid()
    plt.xlim(x1, x2)
    plt.plot(np.transpose(file)[n])
    plt.legend([channels[n]])
    plt.title(sid+' '+eyes.upper()+' '+label.upper()+' '+stat.upper()+' '+ channels[n], size='large')
    plt.xlabel('Seconds')
    plt.ylabel('Sensor Value (uV)')

# plot all 19 channels in seperate graphs
def plot_all_single_channels(file):
    eyes, label, sid, stat = file.lower().split('_')
    file = loadmat(resolve_path(eyes, label, sid, stat))
    if stat=='raw':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    
    trans = np.transpose(file)
    x=0
    for channel_data in trans:
        plt.figure(x, figsize=(16,4))
        plt.grid()
        plt.plot(trans[x])
        plt.legend([channels[x]])
        plt.title(sid+' '+eyes.upper()+' '+label.upper()+' '+stat.upper()+' '+ channels[x], size='large')
        plt.xlabel('Seconds')
        plt.ylabel('Sensor Value (uV)')
        x+=1
        
# 3d plot for all channels for the full 3 mins. optional arg for color map.
def plot_3d(file,cm='magma'):
    eyes, label, sid, stat = file.lower().split('_')
    if stat=='raw':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    x = np.arange(19)
    y = np.arange(256*180)
    z = np.array(file)

    xx, yy = np.meshgrid(x, y)

    ax.plot_surface(xx, yy, z, cmap=cm)

    ax.set_title(sid+' '+eyes.upper()+' '+label.upper()+' '+stat.upper(), size='large')
    ax.set_xlabel('Channel', size='large')
    ax.set_xticks(np.linspace(1, 19, num=19, endpoint=True))
    ax.set_ylabel('Seconds', size='large')
    ax.set_zlabel('Sensor Value', size='large')

    plt.show()

    
"INFORMATION FUNCTIONS"

# prints the names and numbers of channels
def channel_names():
    for i in range(len(channels)):
        print(f'{channels[i]}: {i+1}')

    
"FILE DATA RETRIEVAL FUNCTIONS"    
    
# returns the matfile data as a (256x180) x 19 list
def get_data_as_list(file):
    # format: EC_A_103_RAW
    eyes, label, sid, stat = file.lower().split('_')
    if stat=='raw':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    return file
    
# returns the matfile data as a (256x180) x 19 numpy array
def get_data_as_np_array(file):
    # format: EC_A_103_RAW
    eyes, label, sid, stat = file.lower().split('_')
    if stat=='raw':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    return np.array(file)
 
# returns the data for a single channel as an np array   
def get_single_channel_data(file, channel_name='', channel_no=0):
    eyes, label, sid, stat = file.lower().split('_')
    if stat=='raw':
        data = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        data = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        data = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    if channel_name=='':
        return data[channel_no-1]
    else:
        for i in range(len(channels)):
            if channels[i].lower()==channel_name.lower():
                return data[i]
        
# returns the matfile data only for the specified channels, as dict of numpy arrays
# kwargs: channel_names, channel_nos
def get_channel_data(file, **chans):
    
    eyes, label, sid, stat = file.lower().split('_')
    if stat=='raw':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    data = {}
    
    chan_names = [c.lower() for c in channels]
    
    if 'channel_names' in chans:
        for c in chans['channel_names']:
            data[c] = file[chan_names.index(c.lower())]
            
    if 'channel_nos' in chans:
        for c in chans['channel_nos']:
            data[c] = file[c-1]
        
    return data

# returns the matfile data as a pandas dataframe
def get_data_as_pd_dataframe(file):
    eyes, label, sid, stat = file.lower().split('_')
    if stat=='raw':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    df = pd.DataFrame(file)
    df.set_axis(channels, axis=1, inplace=True)
    
    return df
        
"SPECTROGRAMS"

# default args: 
FS=256
CM='jet'
N=512
NOVERLAP=92
COLORBAR=False

# plot spectrogram for the specified channel w the default args. (or specify vals)
def spec_single_channel(file, channel_name='', channel_no=0, fs=FS, cm=CM, nfft=N, noverlap=NOVERLAP, show_colorbar=False):
    eyes, label, sid, stat = file.lower().split('_')
    if stat=='raw':
        data = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        data = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        data = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    
    if channel_name=='':
        data = data[channel_no-1]
        c=channel_no
    else:
        for i in range(len(channels)):
            if channels[i].lower()==channel_name.lower():
                data = data[i]
                c=channel_name
    
    plt.specgram(x=data, Fs=fs, cmap=cm, NFFT=nfft, noverlap=noverlap)
    plt.title(f'Channel: {c}    {eyes} {label} {sid} {stat}')
    plt.show()
    
    
# plot spectrogram for the specified channel(s) w the default args. (or specify vals)
def spec_for_channels(file, channel_names=[],channel_nos=[], fs=FS, cm=CM, nfft=N, noverlap=NOVERLAP, show_colorbar=False):
    eyes, label, sid, stat = file.lower().split('_')
    if stat=='raw':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    
    chan_names = [c.lower() for c in channels]
    
    if channel_names!=[]:
        for c in chans['channel_names']:
            data = file[chan_names.index(c.lower())]
            plt.figure()
            plt.title(f'Channel: {c}    {eyes} {label} {sid} {stat}')
            plt.specgram(x=data, Fs=fs, cmap=cm, NFFT=nfft, noverlap=noverlap)
            plt.show()
            
    if channel_nos!=[]:
        for c in chans['channel_nos']:
            data = file[c-1]
            plt.figure()
            plt.title(f'Channel: {c} ({channels[c-1]})    {eyes} {label} {sid} {stat}')
            plt.specgram(x=data, Fs=fs, cmap=cm, NFFT=nfft, noverlap=noverlap)
            plt.show()

# plot spectrogram for all 19 channels of the file w the default args. (or specify vals)
def specs_for_file(file, fs=FS, cm=CM, nfft=N, noverlap=NOVERLAP, show_colorbar=False):
    eyes, label, sid, stat = file.lower().split('_')
    if stat=='raw':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data']).T
    elif stat=='aar':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))['data'])
    elif stat=='adj':
        file = np.array(loadmat(resolve_path(eyes, label, sid, stat))[eyes.upper()+'_Data'])
    
    i=1
    for f in file:
        plt.specgram(x=f, Fs=fs, cmap=cm, NFFT=nfft, noverlap=noverlap)
        plt.title(f'Channel: {i} ({channels[i-1]})    {eyes} {label} {sid} {stat}')
        plt.show()
        i+=1