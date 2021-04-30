#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#CHINMAY SANJAY WARANG
#DST LAB FINAL PROJECT
#GRANULAR SYNTHESIZER

#importing libraries

import pyaudio,wave
import struct
import math
import tkinter as Tk
import numpy as np
import threading


#Defining Functions

#For increasing the Speed at which the grains are played

def fun_up():  
    global RATE
    if RATE >= 128000:
        RATE = RATE
        print(RATE)
    else:
        RATE = RATE*2 
        
#For decreasing the Speed at which the grains are played
        
def fun_down():
    global RATE
    RATE = RATE/4
    
#For starting the stream
    
def fun_play():
    global CONTINUE
    global my_thread

    if not CONTINUE:
        CONTINUE = True
        my_thread = threading.Thread(target=main)
        my_thread.start()
        
#For stopping the stream
    
def fun_quit():
    global CONTINUE
    global my_thread

    if CONTINUE:
        CONTINUE = False
        print('See You Again')
        my_thread.join()
        
#Main Code
        
def main():
    
    global CONTINUE
    global my_thread
    global RATE
    global f0
    global BLOCKLEN
    global WIDTH
    global wet
    global dry
    global master
    
    while CONTINUE:
    

        BLOCKLEN = 512
        MAXVALUE = 2**15-1
        wavfile = 'Guitar.wav'
        output_wavfile = 'New Grain Track.wav'
        
        # Open wave file 
        
        wf = wave.open( wavfile, 'rb' )
        
        
        CHANNELS        = wf.getnchannels()     # Number of channels
        RATE            = wf.getframerate()     # Sampling rate (frames/second)
        LEN             = wf.getnframes()       # Signal length
        WIDTH           = wf.getsampwidth()     # Number of bytes per sample
        
        # Setting parameters for writing the wave file
        
        output_wf = wave.open(output_wavfile, 'w')     
        output_wf.setframerate(RATE)
        output_wf.setsampwidth(WIDTH)
        output_wf.setnchannels(CHANNELS)

        # Open the audio stream
        p = pyaudio.PyAudio()
        
        PA_FORMAT = p.get_format_from_width(WIDTH)
        
        stream = p.open(
            format = PA_FORMAT,
            channels = CHANNELS,
            rate = RATE,
            input = True,
            output = True)


        # Input Parameters
        theta = 0
        output_block_1 = BLOCKLEN * [0]
        start_gain_1 = 0
        start_gain_2 = 0
        start_gain = 0
        
        # Read audio input stream
        
        input_bytes = wf.readframes(BLOCKLEN)


        while len(input_bytes) == WIDTH * BLOCKLEN:
            root.update()
            
            om = 2*math.pi*f0.get()/RATE
            
            #final gain values 
            
            final_gain_1 = wet.get() 
            final_gain_2 = dry.get()
            final_gain = master.get()
            
            #update values
            
            update_1 = np.linspace(start_gain_1,final_gain_1,BLOCKLEN)
            update_2 = np.linspace(start_gain_2,final_gain_2,BLOCKLEN)
            update = np.linspace(start_gain,final_gain,BLOCKLEN)
            
            #unpacking 
            
            signal_block = struct.unpack('h' * BLOCKLEN, input_bytes)  
            
            #Modulating Grains
            
            for n in range(0, BLOCKLEN):
                theta = theta + om
                output_block_1[n] = int( signal_block[n] * math.cos(theta) )
                
            #Output Stream
            
            output_block = output_block_1*update_1+signal_block*update_2
            output_block = output_block*update
            while theta > math.pi:
                theta = theta - 2*math.pi
            output_block = output_block.astype(int)
            output_block = np.clip(output_block, -MAXVALUE, MAXVALUE)
            output_bytes = struct.pack('h' * BLOCKLEN, *output_block)
            stream.write(output_bytes)
            
            #Writing The Audio 
            
            output_wf.writeframes(output_bytes)
            
            #Updating the start gain to the previous value of gain
            
            start_gain = final_gain  
            start_gain_1 = final_gain_1 
            start_gain_2 = final_gain_2 
            
            
            #Next Frame
            
            input_bytes = wf.readframes(BLOCKLEN)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf.close()
        output_wf.close()

        print('Finished Playing Audio File')
        
CONTINUE = False
my_thread = None


# Define Tkinter root
root = Tk.Tk()

# Define Tk variables
wet = Tk.DoubleVar()
dry = Tk.DoubleVar()
master = Tk.DoubleVar()
f0 = Tk.DoubleVar()

# Initialize Tk variables
wet.set(0)
dry.set(50)
f0.set(400)
master.set(40)

# Define widgets
L1 = Tk.Label(root, text = 'REAL TIME GRANULAR SYNTHSIZER')
B_up = Tk.Button(root, text = 'Speed Up',command = fun_up)
B_down = Tk.Button(root, text = 'Slow Down', command = fun_down)
S_wet = Tk.Scale(root, label = 'Wet', variable = wet, from_ = 0, to = 100)
S_dry = Tk.Scale(root, label = 'Dry', variable = dry, from_ = 0, to = 100)
S_master = Tk.Scale(root, label = 'Master Volume', variable = master, from_ = 0, to = 100)
S_freq = Tk.Scale(root, label = 'Frequency', variable = f0, from_ = 50, to = 1000)
B_quit = Tk.Button(root, text = 'Quit', command = fun_quit)
B_play = Tk.Button(root, text = 'Play', command = fun_play)


# Place widgets
L1.pack()
B_quit.pack(side = Tk.BOTTOM,fill = Tk.BOTH)
B_play.pack(side = Tk.TOP,fill = Tk.BOTH)
B_up.pack(side = Tk.LEFT,fill = Tk.Y)
B_down.pack(side = Tk.RIGHT,fill = Tk.Y)
S_wet.pack(side = Tk.LEFT,fill = Tk.BOTH)
S_dry.pack(side = Tk.LEFT,fill = Tk.BOTH)
S_freq.pack(side = Tk.LEFT,fill = Tk.BOTH)
S_master.pack(side = Tk.RIGHT,fill = Tk.BOTH)

root.mainloop()



