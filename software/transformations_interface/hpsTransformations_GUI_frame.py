# GUI frame for the hpsTransformations_function.py

from Tkinter import *
import tkFileDialog, tkMessageBox
import sys, os
from scipy.io.wavfile import read
import numpy as np
import hpsTransformations_function as hT
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import json
import traceback
 
class HpsTransformations_frame:
  
    def __init__(self, parent):  
        self.parent = parent
        self.isStereoMode = False       
        self.initUI(createWidgets=True)
        self.initUI(createWidgets=False)

    def toggleStereoMode(self):
        self.isStereoMode = not self.isStereoMode
        for widget in self.parent.winfo_children():
            widget.grid_forget()
        self.initUI(createWidgets=False)

    def initUI(self, createWidgets):

        if createWidgets: 
            self.choose_label_text = "inputFile:"            
            self.choose_label = Label(self.parent, text=self.choose_label_text)
        else:
            self.choose_label.grid(row=0, column=0, sticky=W, padx=5, pady=(10,2))
 
        #TEXTBOX TO PRINT PATH OF THE SOUND FILE
        if createWidgets:
            self.filelocation_default = '../../sounds/sax-phrase-short.wav'
            self.filelocation = Entry(self.parent)
            self.filelocation.focus_set()
            self.filelocation["width"] = 32
            self.filelocation.delete(0, END)
            self.filelocation.insert(0, self.filelocation_default)
        else:
            self.filelocation.grid(row=0,column=0, sticky=W, padx=(70, 5), pady=(10,2))

        #BUTTON TO BROWSE SOUND FILE
        if createWidgets: self.open_file_button = Button(self.parent, text="...", command=self.browse_file) #see: def browse_file(self)
        else: self.open_file_button.grid(row=0, column=0, sticky=W, padx=(340, 6), pady=(10,2)) #put it beside the filelocation textbox
 
        #BUTTON TO PREVIEW SOUND FILE
        if createWidgets: self.preview_button = Button(self.parent, text=">", command=lambda:UF.wavplay(self.filelocation.get()), bg="gray30", fg="white")
        else: self.preview_button.grid(row=0, column=0, sticky=W, padx=(385,6), pady=(10,2))

        ## HPS TRANSFORMATIONS ANALYSIS

        #ANALYSIS WINDOW TYPE
        if createWidgets:
            self.wtype_label = Label(self.parent, text="window:")
            self.w_type = StringVar()
            self.w_type.set("blackman") # initial value
            self.window_option = OptionMenu(self.parent, self.w_type, "rectangular", "hanning", "hamming", "blackman", "blackmanharris")
        else:
            self.wtype_label.grid(row=1, column=0, sticky=W, padx=5, pady=(10,2))
            self.window_option.grid(row=1, column=0, sticky=W, padx=(65,5), pady=(10,2))

        #WINDOW SIZE
        if createWidgets:
            self.M_label_text = "M:"
            self.M_label = Label(self.parent, text=self.M_label_text)
            self.M = Entry(self.parent, justify=CENTER)
            self.M["width"] = 5
            self.M.delete(0, END)
            self.M.insert(0, "601")
        else:
            self.M_label.grid(row=1, column=0, sticky=W, padx=(180, 5), pady=(10,2))
            self.M.grid(row=1,column=0, sticky=W, padx=(200,5), pady=(10,2))

        #FFT SIZE
        if createWidgets:
            self.N_label_text = "N:"
            self.N_label = Label(self.parent, text=self.N_label_text)
            self.N = Entry(self.parent, justify=CENTER)
            self.N["width"] = 5
            self.N.delete(0, END)
            self.N.insert(0, "1024")
        else:
            self.N_label.grid(row=1, column=0, sticky=W, padx=(255, 5), pady=(10,2))
            self.N.grid(row=1,column=0, sticky=W, padx=(275,5), pady=(10,2))

        #THRESHOLD MAGNITUDE
        if createWidgets:
            self.t_label_text = "t:"
            self.t_label = Label(self.parent, text=self.t_label_text)
            self.t = Entry(self.parent, justify=CENTER)
            self.t["width"] = 5
            self.t.delete(0, END)
            self.t.insert(0, "-100")
        else:
            self.t_label.grid(row=1, column=0, sticky=W, padx=(330,5), pady=(10,2))
            self.t.grid(row=1, column=0, sticky=W, padx=(348,5), pady=(10,2))

        #MIN DURATION SINUSOIDAL TRACKS
        if createWidgets:
            self.minSineDur_label_text = "minSineDur:"
            self.minSineDur_label = Label(self.parent, text=self.minSineDur_label_text)
            self.minSineDur = Entry(self.parent, justify=CENTER)
            self.minSineDur["width"] = 5
            self.minSineDur.delete(0, END)
            self.minSineDur.insert(0, "0.1")
        else:
            self.minSineDur_label.grid(row=2, column=0, sticky=W, padx=(5, 5), pady=(10,2))
            self.minSineDur.grid(row=2, column=0, sticky=W, padx=(87,5), pady=(10,2))

        #MAX NUMBER OF HARMONICS
        if createWidgets:
            self.nH_label_text = "nH:"
            self.nH_label = Label(self.parent, text=self.nH_label_text)
            self.nH = Entry(self.parent, justify=CENTER)
            self.nH["width"] = 5
            self.nH.delete(0, END)
            self.nH.insert(0, "100")
        else:
            self.nH_label.grid(row=2, column=0, sticky=W, padx=(145,5), pady=(10,2))
            self.nH.grid(row=2, column=0, sticky=W, padx=(172,5), pady=(10,2))

        #MIN FUNDAMENTAL FREQUENCY
        if createWidgets:
            self.minf0_label_text = "minf0:"
            self.minf0_label = Label(self.parent, text=self.minf0_label_text)
            self.minf0 = Entry(self.parent, justify=CENTER)
            self.minf0["width"] = 5
            self.minf0.delete(0, END)
            self.minf0.insert(0, "350")
        else:
            self.minf0_label.grid(row=2, column=0, sticky=W, padx=(227,5), pady=(10,2))
            self.minf0.grid(row=2, column=0, sticky=W, padx=(275,5), pady=(10,2))

        #MAX FUNDAMENTAL FREQUENCY
        if createWidgets:
            self.maxf0_label_text = "maxf0:"
            self.maxf0_label = Label(self.parent, text=self.maxf0_label_text)
            self.maxf0 = Entry(self.parent, justify=CENTER)
            self.maxf0["width"] = 5
            self.maxf0.delete(0, END)
            self.maxf0.insert(0, "700")
        else:
            self.maxf0_label.grid(row=2, column=0, sticky=W, padx=(330,5), pady=(10,2))
            self.maxf0.grid(row=2, column=0, sticky=W, padx=(380,5), pady=(10,2))

        #MAX ERROR ACCEPTED
        if createWidgets:
            self.f0et_label_text = "f0et:"
            self.f0et_label = Label(self.parent, text=self.f0et_label_text)
            self.f0et = Entry(self.parent, justify=CENTER)
            self.f0et["width"] = 3
            self.f0et.delete(0, END)
            self.f0et.insert(0, "7")
        else:
            self.f0et_label.grid(row=3, column=0, sticky=W, padx=5, pady=(10,2))
            self.f0et.grid(row=3, column=0, sticky=W, padx=(42,5), pady=(10,2))

        #ALLOWED DEVIATION OF HARMONIC TRACKS
        if createWidgets:
            self.harmDevSlope_label_text = "harmDevSlope:"
            self.harmDevSlope_label = Label(self.parent, text=self.harmDevSlope_label_text)
            self.harmDevSlope = Entry(self.parent, justify=CENTER)
            self.harmDevSlope["width"] = 5
            self.harmDevSlope.delete(0, END)
            self.harmDevSlope.insert(0, "0.01")
        else:
            self.harmDevSlope_label.grid(row=3, column=0, sticky=W, padx=(90,5), pady=(10,2))
            self.harmDevSlope.grid(row=3, column=0, sticky=W, padx=(190,5), pady=(10,2))

        #DECIMATION FACTOR
        if createWidgets:
            self.stocf_label_text = "stocf:"
            self.stocf_label = Label(self.parent, text=self.stocf_label_text)
            self.stocf = Entry(self.parent, justify=CENTER)
            self.stocf["width"] = 5
            self.stocf.delete(0, END)
            self.stocf.insert(0, "0.1")
        else:
            self.stocf_label.grid(row=3, column=0, sticky=W, padx=(250,5), pady=(10,2))
            self.stocf.grid(row=3, column=0, sticky=W, padx=(290,5), pady=(10,2))

        #BUTTON TO DO THE ANALYSIS OF THE SOUND
        if createWidgets: self.computeAnalysisSynthesisButton = Button(self.parent, text="Analysis/Synthesis", command=self.analysis, bg="dark red", fg="white")
        else: self.computeAnalysisSynthesisButton.grid(row=4, column=0, padx=5, pady=(10,5), sticky=W)
        
        #BUTTON TO PLAY ANALYSIS/SYNTHESIS OUTPUT
        if createWidgets: self.playAnalysisSynthesisButton = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hpsModel.wav'), bg="gray30", fg="white")
        else: self.playAnalysisSynthesisButton.grid(row=4, column=0, padx=(145,5), pady=(10,5), sticky=W)

        #BUTTON TO TOGGLE STEREO MODE
        if createWidgets: self.toggleStereoModeButton = Button(self.parent, text="Toggle Stereo Transformation Mode", command=self.toggleStereoMode, bg="dark blue", fg="white")
        else: self.toggleStereoModeButton.grid(row=4, column=0, padx=5, pady=(10,5), sticky=E)

        #SEPARATION LINE
        if createWidgets: self.separator_1 = Frame(self.parent,height=1,width=50,bg="black")
        else: self.separator_1.grid(row=5, pady=5, sticky=W+E)
        ###

        if self.isStereoMode or createWidgets:
            
            #FREQUENCY SCALING FACTORS (L)
            if createWidgets:
                self.freqScaling_label_left_text = "Frequency scaling factors for Left Channel (time, value pairs):"
                self.freqScaling_label_left = Label(self.parent, text=self.freqScaling_label_left_text)
                self.freqScalingLeft = Entry(self.parent, justify=CENTER)
                self.freqScalingLeft["width"] = 35
                self.freqScalingLeft.delete(0, END)
                self.freqScalingLeft.insert(0, "[0, 1.2, 2.01, 1.2, 2.679, .9, 3.146, .9]")
            else:    
                self.freqScaling_label_left.grid(row=6, column=0, sticky=W, padx=5, pady=(5,2))
                self.freqScalingLeft.grid(row=7, column=0, sticky=W+E, padx=5, pady=(0,2))
    
            #FREQUENCY SCALING FACTORS (R)
            if createWidgets:
                self.freqScaling_label_right_text = "Frequency scaling factors for Right Channel (time, value pairs):"
                self.freqScaling_label_right = Label(self.parent, text=self.freqScaling_label_right_text)
                self.freqScalingRight = Entry(self.parent, justify=CENTER)
                self.freqScalingRight["width"] = 35
                self.freqScalingRight.delete(0, END)
                self.freqScalingRight.insert(0, "[0, 1.2, 2.01, 1.25, 2.679, .7, 3.146, .7]")
            else:    
                self.freqScaling_label_right.grid(row=8, column=0, sticky=W, padx=5, pady=(5,2))
                self.freqScalingRight.grid(row=9, column=0, sticky=W+E, padx=5, pady=(0,2))
    
            #FREQUENCY STRETCHING FACTORS (L)
            if createWidgets:
                self.freqStretching_label_left_text = "Frequency stretching factors for Left Channel (time, value pairs):"
                self.freqStretching_label_left = Label(self.parent, text=self.freqStretching_label_left_text)
                self.freqStretchingLeft = Entry(self.parent, justify=CENTER)
                self.freqStretchingLeft["width"] = 35
                self.freqStretchingLeft.delete(0, END)
                self.freqStretchingLeft.insert(0, "[0, 1, 2.01, 1, 2.679, 1.3, 3.146, 1.3]")
            else:    
                self.freqStretching_label_left.grid(row=10, column=0, sticky=W, padx=5, pady=(5,2))
                self.freqStretchingLeft.grid(row=11, column=0, sticky=W+E, padx=5, pady=(0,2))
   
            #FREQUENCY STRETCHING FACTORS (R)
            if createWidgets:
                self.freqStretching_label_right_text = "Frequency stretching factors for Right Channel (time, value pairs):"
                self.freqStretching_label_right = Label(self.parent, text=self.freqStretching_label_right_text)
                self.freqStretchingRight = Entry(self.parent, justify=CENTER)
                self.freqStretchingRight["width"] = 35
                self.freqStretchingRight.delete(0, END)
                self.freqStretchingRight.insert(0, "[0, 1, 2.01, 1, 2.679, 1.5, 3.146, 1.5]")
            else:    
                self.freqStretching_label_right.grid(row=12, column=0, sticky=W, padx=5, pady=(5,2))
                self.freqStretchingRight.grid(row=13, column=0, sticky=W+E, padx=5, pady=(0,2))
    
            #TIMBRE PRESERVATION (L)
            if createWidgets:
                self.timbrePreservation_label_left_text = "Timbre preservation L (1 preserves original timbre, 0 does not):"
                self.timbrePreservation_label_left = Label(self.parent, text=self.timbrePreservation_label_left_text)
                self.timbrePreservationLeft = Entry(self.parent, justify=CENTER)
                self.timbrePreservationLeft["width"] = 2
                self.timbrePreservationLeft.delete(0, END)
                self.timbrePreservationLeft.insert(0, "1")
            else:    
                self.timbrePreservation_label_left.grid(row=14, column=0, sticky=W, padx=5, pady=(5,2))
                self.timbrePreservationLeft.grid(row=14, column=0, sticky=W+E, padx=(395,5), pady=(5,2))

            #TIMBRE PRESERVATION (R)
            if createWidgets:
                self.timbrePreservation_label_right_text = "Timbre preservation R (1 preserves original timbre, 0 does not):"
                self.timbrePreservation_label_right = Label(self.parent, text=self.timbrePreservation_label_right_text)
                self.timbrePreservationRight = Entry(self.parent, justify=CENTER)
                self.timbrePreservationRight["width"] = 2
                self.timbrePreservationRight.delete(0, END)
                self.timbrePreservationRight.insert(0, "1")
            else:    
                self.timbrePreservation_label_right.grid(row=15, column=0, sticky=W, padx=5, pady=(5,2))
                self.timbrePreservationRight.grid(row=15, column=0, sticky=W+E, padx=(395,5), pady=(5,2))
    
            #TIME SCALING FACTORS (L)
            if createWidgets:
                self.timeScaling_label_left_text = "Time scaling factors  for Left Channel (time, value pairs):"
                self.timeScaling_label_left = Label(self.parent, text=self.timeScaling_label_left_text)
                self.timeScalingLeft = Entry(self.parent, justify=CENTER)
                self.timeScalingLeft["width"] = 35
                self.timeScalingLeft.delete(0, END)
                self.timeScalingLeft.insert(0, "[0, 0, 2.138, 2.138-1.0, 3.146, 3.146]")
            else:    
                self.timeScaling_label_left.grid(row=16, column=0, sticky=W, padx=5, pady=(5,2))
                self.timeScalingLeft.grid(row=17, column=0, sticky=W+E, padx=5, pady=(0,2))
    
            #TIME SCALING FACTORS (R)
            if createWidgets:
                self.timeScaling_label_right_text = "Time scaling factors  for Right Channel (time, value pairs):"
                self.timeScaling_label_right = Label(self.parent, text=self.timeScaling_label_right_text)
                self.timeScalingRight = Entry(self.parent, justify=CENTER)
                self.timeScalingRight["width"] = 35
                self.timeScalingRight.delete(0, END)
                self.timeScalingRight.insert(0, "[0, 0, 2.138, 2.138-1.0, 3.146, 3.146]")
            else:    
                self.timeScaling_label_right.grid(row=18, column=0, sticky=W, padx=5, pady=(5,2))
                self.timeScalingRight.grid(row=19, column=0, sticky=W+E, padx=5, pady=(0,2))

        if (not self.isStereoMode) or createWidgets:
          
            #FREQUENCY SCALING FACTORS
            if createWidgets:
                self.freqScaling_label_text = "Frequency scaling factors (time, value pairs):"
                self.freqScaling_label = Label(self.parent, text=self.freqScaling_label_text)
                self.freqScaling = Entry(self.parent, justify=CENTER)
                self.freqScaling["width"] = 35
                self.freqScaling.delete(0, END)
                self.freqScaling.insert(0, "[0, 1.2, 2.01, 1.2, 2.679, .7, 3.146, .7]")
            else:    
                self.freqScaling_label.grid(row=6, column=0, sticky=W, padx=5, pady=(5,2))
                self.freqScaling.grid(row=7, column=0, sticky=W+E, padx=5, pady=(0,2))
    
            #FREQUENCY STRETCHING FACTORS
            if createWidgets:
                self.freqStretching_label_text = "Frequency stretching factors (time, value pairs):"
                self.freqStretching_label = Label(self.parent, text=self.freqStretching_label_text)
                self.freqStretching = Entry(self.parent, justify=CENTER)
                self.freqStretching["width"] = 35
                self.freqStretching.delete(0, END)
                self.freqStretching.insert(0, "[0, 1, 2.01, 1, 2.679, 1.5, 3.146, 1.5]")
            else:    
                self.freqStretching_label.grid(row=8, column=0, sticky=W, padx=5, pady=(5,2))
                self.freqStretching.grid(row=9, column=0, sticky=W+E, padx=5, pady=(0,2))
    
            #TIMBRE PRESERVATION
            if createWidgets:
                self.timbrePreservation_label_text = "Timbre preservation (1 preserves original timbre, 0 it does not):"
                self.timbrePreservation_label = Label(self.parent, text=self.timbrePreservation_label_text)
                self.timbrePreservation = Entry(self.parent, justify=CENTER)
                self.timbrePreservation["width"] = 2
                self.timbrePreservation.delete(0, END)
                self.timbrePreservation.insert(0, "1")
            else:    
                self.timbrePreservation_label.grid(row=10, column=0, sticky=W, padx=5, pady=(5,2))
                self.timbrePreservation.grid(row=10, column=0, sticky=W+E, padx=(395,5), pady=(5,2))
    
            #TIME SCALING FACTORS
            if createWidgets:
                self.timeScaling_label_text = "Time scaling factors (time, value pairs):"
                self.timeScaling_label = Label(self.parent, text=self.timeScaling_label_text)
                self.timeScaling = Entry(self.parent, justify=CENTER)
                self.timeScaling["width"] = 35
                self.timeScaling.delete(0, END)
                self.timeScaling.insert(0, "[0, 0, 2.138, 2.138-1.0, 3.146, 3.146]")
            else:    
                self.timeScaling_label.grid(row=11, column=0, sticky=W, padx=5, pady=(5,2))
                self.timeScaling.grid(row=12, column=0, sticky=W+E, padx=5, pady=(0,2))
    
        #BUTTON TO APPLY THE TRANSFORMATION
        if createWidgets: self.applyTransformationButton = Button(self.parent, text="Apply Transformation", command=self.transformation_synthesis, bg="dark green", fg="white")
        else: self.applyTransformationButton.grid(row=20, column=0, padx=5, pady=(10,15), sticky=W)

        #BUTTON TO PLAY TRANSFORMATION SYNTHESIS OUTPUT
        if createWidgets:self.outputTransformationButton = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hpsModelTransformation.wav'), bg="gray30", fg="white")
        else: self.outputTransformationButton.grid(row=20, column=0, padx=(164,5), pady=(10,15), sticky=W)

        #BUTTON TO SAVE PARAMETERS
        if createWidgets: self.saveParametersButton = Button(self.parent, text="Save Config", command=self.save_parameters, bg="dark blue", fg="white")
        else: self.saveParametersButton.grid(row=20, column=0, padx=(5,106), pady=(10,15), sticky=E)
 
        #BUTTON TO LOAD PARAMETERS
        if createWidgets: self.loadParametersButton = Button(self.parent, text="Load Config", command=self.load_parameters, bg="dark blue", fg="white")
        else: self.loadParametersButton.grid(row=20, column=0, padx=5, pady=(10,15), sticky=E)
        
        # define options for opening files
        if createWidgets:
            self.file_opt = {}
            self.file_opt['defaultextension'] = '.wav'
            self.file_opt['filetypes'] = [('Wav files', '.wav'),('All files', '.*')]
            self.file_opt['initialdir'] = '../../sounds/'
            self.file_opt['title'] = 'Open a mono audio file .wav with sample frequency 44100 Hz'
            self.pars_opt = {}
            self.pars_opt['defaultextension'] = '.hps'
            self.pars_opt['filetypes'] = [('HPS parameters files', '.hps'),('All files', '.*')]
            self.pars_opt['initialdir'] = 'output_sounds/'
            self.pars_opt['title'] = 'Open a file .hps with transformation parameters'

    def browse_file(self):
        
        filename = tkFileDialog.askopenfilename(**self.file_opt)
 
        #set the text of the self.filelocation
        self.filelocation.delete(0, END)
        self.filelocation.insert(0,filename)
        
    def save_parameters(self):
        out = tkFileDialog.asksaveasfile(mode='w', **self.pars_opt)
        params = {
            '01 filelocation'           : self.filelocation.get(),
            '02 w_type'                 : self.w_type.get(),
            '03 M'                      : self.M.get(),
            '04 N'                      : self.N.get(),
            '05 t'                      : self.t.get(),
            '06 minSineDur'             : self.minSineDur.get(),
            '07 nH'                     : self.nH.get(),
            '08 minf0'                  : self.minf0.get(),
            '09 maxf0'                  : self.maxf0.get(),
            '10 f0et'                   : self.f0et.get(),
            '11 harmDevSlope'           : self.harmDevSlope.get(),
            '12 stocf'                  : self.stocf.get(),
            
            '13 freqScaling'            : self.freqScaling.get(),
            '14 freqStretching'         : self.freqStretching.get(),
            '15 timbrePreservation'     : self.timbrePreservation.get(),
            '16 timeScaling'            : self.timeScaling.get(),

            '17 freqScalingLeft'        : self.freqScalingLeft.get(),
            '18 freqScalingRight'       : self.freqScalingRight.get(),
            '19 freqStretchingLeft'     : self.freqStretchingLeft.get(),
            '20 freqStretchingRight'    : self.freqStretchingRight.get(),
            '21 timbrePreservationLeft' : self.timbrePreservationLeft.get(),
            '22 timbrePreservationRight': self.timbrePreservationRight.get(),
            '23 timeScalingLeft'        : self.timeScalingLeft.get(),
            '24 timeScalingRight'       : self.timeScalingRight.get()
        }
        json.dump(params,out,sort_keys=True,indent=0)
        out.close()

    def load_parameters(self):
        inp = tkFileDialog.askopenfile(mode='r', **self.pars_opt)
        params = json.load(inp)
        inp.close()
        
        def set_entry(entry,newValue):
            entry.delete(0,END)
            entry.insert(0,str(newValue))
                   
        set_entry(self.filelocation, params           ['01 filelocation'])
        self.w_type.set(str(params                    ['02 w_type']))
        set_entry(self.M, params                      ['03 M'])
        set_entry(self.N, params                      ['04 N'])
        set_entry(self.t, params                      ['05 t'])
        set_entry(self.minSineDur, params             ['06 minSineDur'])
        set_entry(self.nH, params                     ['07 nH'])
        set_entry(self.minf0, params                  ['08 minf0'])
        set_entry(self.maxf0, params                  ['09 maxf0'])
        set_entry(self.f0et, params                   ['10 f0et'])
        set_entry(self.harmDevSlope, params           ['11 harmDevSlope'])
        set_entry(self.stocf, params                  ['12 stocf'])
            
        set_entry(self.freqScaling, params            ['13 freqScaling'])
        set_entry(self.freqStretching, params         ['14 freqStretching'])
        set_entry(self.timbrePreservation, params     ['15 timbrePreservation'])
        set_entry(self.timeScaling, params            ['16 timeScaling'])

        set_entry(self.freqScalingLeft, params        ['17 freqScalingLeft'])
        set_entry(self.freqScalingRight, params       ['18 freqScalingRight'])
        set_entry(self.freqStretchingLeft, params     ['19 freqStretchingLeft'])
        set_entry(self.freqStretchingRight, params    ['20 freqStretchingRight'])
        set_entry(self.timbrePreservationLeft, params ['21 timbrePreservationLeft'])
        set_entry(self.timbrePreservationRight, params['22 timbrePreservationRight'])
        set_entry(self.timeScalingLeft, params        ['23 timeScalingLeft'])
        set_entry(self.timeScalingRight, params       ['24 timeScalingRight'])

    def analysis(self):
        
        try:
            inputFile = self.filelocation.get()
            window = self.w_type.get()
            M = int(self.M.get())
            N = int(self.N.get())
            t = int(self.t.get())
            minSineDur = float(self.minSineDur.get())
            nH = int(self.nH.get())
            minf0 = int(self.minf0.get())
            maxf0 = int(self.maxf0.get())
            f0et = int(self.f0et.get())
            harmDevSlope = float(self.harmDevSlope.get())
            stocf = float(self.stocf.get())
 
            self.inputFile, self.fs, self.hfreq, self.hmag, self.mYst, self.inputSound = hT.analysis(inputFile, window, M, N, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope, stocf)

        except ValueError:
            tkMessageBox.showerror("Input values error", "Some parameters are incorrect")
        except Exception as e:
            tkMessageBox.showerror("Analysis error",str(e) + "\n" + traceback.format_exc())   
            

    def transformation_synthesis(self):

        try:
            inputFile = self.filelocation.get()
            fs =  self.fs
            hfreq = self.hfreq
            hmag = self.hmag
            mYst = self.mYst

            if self.isStereoMode:
                
                freqScaling        = (np.array(eval(self.freqScalingLeft.get())),    np.array(eval(self.freqScalingRight.get())))
                freqStretching     = (np.array(eval(self.freqStretchingLeft.get())), np.array(eval(self.freqStretchingRight.get())))
                timbrePreservation = (int(self.timbrePreservationLeft.get()),        int(self.timbrePreservationRight.get()))
                timeScaling        = (np.array(eval(self.timeScalingLeft.get())),    np.array(eval(self.timeScalingRight.get())))
                
                hT.transformation_synthesis_stereo(inputFile, fs, hfreq, hmag, mYst, freqScaling, freqStretching, timbrePreservation, timeScaling, self.inputSound)
           
            if not self.isStereoMode:
               
                freqScaling = np.array(eval(self.freqScaling.get()))
                freqStretching = np.array(eval(self.freqStretching.get()))
                timbrePreservation = int(self.timbrePreservation.get())
                timeScaling = np.array(eval(self.timeScaling.get()))
               
                hT.transformation_synthesis(inputFile, fs, hfreq, hmag, mYst, freqScaling, freqStretching, timbrePreservation, timeScaling)

        except ValueError as errorMessage:
            tkMessageBox.showerror("Input values error", errorMessage)
        except AttributeError:
            tkMessageBox.showerror("Analysis not computed", "First you must analyse the sound!")
        except Exception as e:            
            tkMessageBox.showerror("Analysis error",str(e) + "\n" + traceback.format_exc())   

