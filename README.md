# Projects
Master's Thesis (2021)

The researcher developed a system based on machine learning and music information
retrieval to recreate the legendary ‘Wall Of Sound’. The first stage was developing a dataset on
which the system was to be trained. The researcher shortlisted ten songs for this purpose and
recorded, mixed, and produced stems for the dataset. All the data related to the dataset was stored
in a CSV file. This data included information regarding the song title, tempo, and instrument
type. The MFCCs were calculated for each of the stems and store along with the other
information. RT60 values were assigned to each stem based on the DAW plugin reverb values for
the particular stem. The system was trained using the K-Nearest Neighbors regression algorithm.
The regression model was tested to predict the RT60 parameters for input files based on the
trained data. The RT60 coefficients predicted were then used to generate artificial reverb by
convolving the input stems with the predicted length impulse response. The resultant stems were
mixed on Python and stored as 44100 kHz 24-bit Wav files to the disk. The engineer mixed the
stems using Logic X. The mixes generated were loudness normalized to -14db LUFS, as a part of
post-processing and preparing the data for listening tests. The listening test was conducted with
IRB approval on 15 participants. The participants performed the test remotely on their own
devices. The participants were asked to listen and rate the two mixes to the original song. The
test consisted of four songs with two mixes for each song. The null hypothesis for this
experiment was that there would be no significant difference in the perception of the two mixes.
The system performed on par with the engineer in terms of ratings and statical analysis of the
results of the listening test, thus proving the efficacy of the system when compared to an
engineer.

JUCE Distortion Pedal AU Plugin (2021)

Developed a Distortion Pedal Plugin using C++ which can be loaded into a DAW and used to generate artificial distortion in real-time. 
The user can adjust the amount of distortion as well as the master volume.
The files are in the SOURCE folder! 

An Intelligent Dataset Augmentation System using Machine Learning to Encode Recording Error (2020)

Developed a dataset expansion tool to introduce errors typically found in recordings on an existing dataset.
Intelligently introduce error to increase the robustness of the models trained with this dataset.
Used different models – 1) Clean Data 2) Dirty Data 3) Mixed Data to evaluate the data expansion system.

Real-Time Granular Synthesizer (2020)

A GUI developed on Python using Tkinter, Struct, Wav and Pyaudio for real-time granular synthesis.
User controls for changing the speed and frequency at which each sample is played. Slider controls for controlling the amount
of Wet/Dry mix in the output as well as a master volume control.

'The Wall Of Sound'Plugin (2020)

Developed a GUI using MATLAB app designer.
Programmed the noise reduction algorithm and parametric effects to be applied to the input audio stems. The plugin mixed the
stems in mono displaying the final waveform which could be played and saved to the disc of the user.

The Birthday Synthesizer Max MSP (2019)

Developed on MAX MSP using the concepts of audio signal processing and jitter.
The synthesizer modulated a sine wave using additive synthesis and subtractive synthesis. The video displayed on the
synthesizer changes colors according to the velocity of the midi input. Interactive controls for playing programmed birthday
song along with visual effects.
