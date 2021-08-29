import os
from mat4py import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np

"SETTING PATHS"

# raw data paths
raw_base = os.path.normpath("D:\Data Division\Raw Data")
eo_adhd = os.path.join(raw_base, 'Eyes-open', 'ADHD')
ec_adhd = os.path.join(raw_base, 'Eyes-closed', 'ADHD')
eo_cont = os.path.join(raw_base, 'Eyes-open', 'Control')
ec_cont = os.path.join(raw_base, 'Eyes-closed', 'Control')

# channel locations path
locs_path = os.path.normpath("D:\Data Division\Standard-10-20-Cap19.locs")

# processed data paths
proc_base = os.path.normpath("D:\MatLab\\toolbox\eeglab2021.0\eeglab2021.0\data_for_processing\EEG_DATA")
# ADJUST
adj_base = os.path.join(proc_base, "Adjust")
adj_eo_adhd = os.path.join(adj_base, 'ADHD_EO')
adj_ec_adhd = os.path.join(adj_base, 'ADHD_EC')
adj_eo_cont = os.path.join(adj_base, 'CONTROL_EO')
adj_ec_cont = os.path.join(adj_base, 'CONTROL_EC')

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
    print('REQUIRED FILE NAME FORMAT: [ EC/EO ]_[ C/A ]_[ ID ]_[ RAW/ADJ ]\t', 'eg: EC_A_103_RAW, eo_c_144_adj')
    print('--'*50,end='\n'+'--'*50+'\n')
    print('FUNC: toolkit_help()','ARGS: none', 'USE: Display available functions and their uses and the file format reqd.', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: plot_all_in_one(file)','ARGS: file name in reqd. format', 'USE: plot all channel data in one graph, every channel centered at x-axis', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: plot_eeg(file)','ARGS: file name in reqd. format', 'USE: plot all channel data in one graph, seperated: main eeg-like plot', sep='\n', end='\n'+'--'*50+'\n')
    print('FUNC: plot_single_channel(file, num=0, name='')','ARGS: file name in reqd. format, reqd. channel name or number', 'USE: plot just a single channel. pass it\'s name or number as kwarg', sep='\n', end='\n'+'--'*50+'\n')
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

# main visualisation functions

# plot all channel data in one graph, every channel centered at x-axis
def plot_all_in_one(file):
    eyes, label, sid, stat = file.lower().split('_')
    file = loadmat(resolve_path(eyes, label, sid, stat))
    
    plt.figure()
    plt.plot(file[eyes.upper()+'_Data'])
    plt.legend(channels, ncol=2, bbox_to_anchor=(1.0, 1.0))
    plt.title(sid+' '+eyes.upper()+' '+label.upper()+' '+stat.upper())
    plt.xlabel('Seconds')
    plt.ylabel('Sensor Value (uV)')
    
# plot all channel data in one graph, seperated: main eeg-like plot
def plot_eeg(file):
    eyes, label, sid, stat = file.lower().split('_')
    file = loadmat(resolve_path(eyes, label, sid, stat))
    
    plt.figure(figsize=(20,10), dpi=80)
    x=0
    for data in np.transpose(file[eyes.upper()+'_Data']):
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
    file = loadmat(resolve_path(eyes, label, sid, stat))
    
    if num==0:
        n=channel_name_to_num(name)
    else: n=num-1 # cuz channel idx 0 = 1st channel
    
    plt.plot(np.transpose(file[eyes.upper()+'_Data'])[n])
    plt.legend([channels[n]])
    plt.title(sid+' '+eyes.upper()+' '+label.upper()+' '+stat.upper()+' '+ channels[n], size='large')
    plt.xlabel('Seconds')
    plt.ylabel('Sensor Value (uV)')

# plot all 19 channels in seperate graphs
def plot_all_single_channels(file):
    eyes, label, sid, stat = file.lower().split('_')
    file = loadmat(resolve_path(eyes, label, sid, stat))
    
    trans = np.transpose(file[eyes.upper()+'_Data'])
    x=0
    for channel_data in trans:
        plt.figure(x)
        plt.plot(trans[x])
        plt.legend([channels[x]])
        plt.title(sid+' '+eyes.upper()+' '+label.upper()+' '+stat.upper()+' '+ channels[x], size='large')
        plt.xlabel('Seconds')
        plt.ylabel('Sensor Value (uV)')
        x+=1
        
# 3d plot for all channels for the full 3 mins. optional arg for color map.
def plot_3d(file,cm='magma'):
    eyes, label, sid, stat = file.lower().split('_')
    file = loadmat(resolve_path(eyes, label, sid, stat))
    
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    x = np.arange(19)
    y = np.arange(256*180)
    z = np.array(file[eyes.upper()+'_Data'])

    xx, yy = np.meshgrid(x, y)

    ax.plot_surface(xx, yy, z, cmap=cm)

    ax.set_title(sid+' '+eyes.upper()+' '+label.upper()+' '+stat.upper(), size='large')
    ax.set_xlabel('Channel', size='large')
    ax.set_xticks(np.linspace(1, 19, num=19, endpoint=True))
    ax.set_ylabel('Seconds', size='large')
    ax.set_zlabel('Sensor Value', size='large')

    plt.show()