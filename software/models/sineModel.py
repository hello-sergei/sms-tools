# functions that implement analysis and synthesis of sounds using the Sinusoidal Model
# (for example usage check the examples models_interface)

import numpy as np
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft, fftshift
import math
import dftModel as DFT
import utilFunctions as UF

def sineTracking(pfreq, pmag, pphase, tfreq, freqDevOffset=20, freqDevSlope=0.01):
	"""
	Tracking sinusoids from one frame to the next
	pfreq, pmag, pphase: frequencies and magnitude of current frame
	tfreq: frequencies of incoming tracks from previous frame
	freqDevOffset: minimum frequency deviation at 0Hz 
	freqDevSlope: slope increase of minimum frequency deviation
	returns tfreqn, tmagn, tphasen: frequency, magnitude and phase of tracks
	"""

	tfreqn = np.zeros(tfreq.size)                              # initialize array for output frequencies
	tmagn = np.zeros(tfreq.size)                               # initialize array for output magnitudes
	tphasen = np.zeros(tfreq.size)                             # initialize array for output phases
	pindexes = np.array(np.nonzero(pfreq), dtype=np.int)[0]    # indexes of current peaks
	incomingTracks = np.array(np.nonzero(tfreq), dtype=np.int)[0] # indexes of incoming tracks
	newTracks = np.zeros(tfreq.size, dtype=np.int) -1           # initialize to -1 new tracks
	magOrder = np.argsort(-pmag[pindexes])                      # order current peaks by magnitude
	pfreqt = np.copy(pfreq)                                     # copy current peaks to temporary array
	pmagt = np.copy(pmag)                                       # copy current peaks to temporary array
	pphaset = np.copy(pphase)                                   # copy current peaks to temporary array

	# continue incoming tracks
	if incomingTracks.size > 0:                                 # if incoming tracks exist
		for i in magOrder:                                        # iterate over current peaks
			if incomingTracks.size == 0:                            # break when no more incoming tracks
				break
			track = np.argmin(abs(pfreqt[i] - tfreq[incomingTracks]))   # closest incoming track to peak
			freqDistance = abs(pfreq[i] - tfreq[incomingTracks[track]]) # measure freq distance
			if freqDistance < (freqDevOffset + freqDevSlope * pfreq[i]):  # choose track if distance is small 
					newTracks[incomingTracks[track]] = i                      # assign peak index to track index
					incomingTracks = np.delete(incomingTracks, track)         # delete index of track in incomming tracks
	indext = np.array(np.nonzero(newTracks != -1), dtype=np.int)[0]   # indexes of assigned tracks
	if indext.size > 0:
		indexp = newTracks[indext]                                    # indexes of assigned peaks
		tfreqn[indext] = pfreqt[indexp]                               # output freq tracks 
		tmagn[indext] = pmagt[indexp]                                 # output mag tracks 
		tphasen[indext] = pphaset[indexp]                             # output phase tracks 
		pfreqt= np.delete(pfreqt, indexp)                             # delete used peaks
		pmagt= np.delete(pmagt, indexp)                               # delete used peaks
		pphaset= np.delete(pphaset, indexp)                           # delete used peaks

	# create new tracks from non used peaks
	emptyt = np.array(np.nonzero(tfreq == 0), dtype=np.int)[0]      # indexes of empty incoming tracks
	peaksleft = np.argsort(-pmagt)                                  # sort left peaks by magnitude
	if ((peaksleft.size > 0) & (emptyt.size >= peaksleft.size)):    # fill empty tracks
			tfreqn[emptyt[:peaksleft.size]] = pfreqt[peaksleft]
			tmagn[emptyt[:peaksleft.size]] = pmagt[peaksleft]
			tphasen[emptyt[:peaksleft.size]] = pphaset[peaksleft]
	elif ((peaksleft.size > 0) & (emptyt.size < peaksleft.size)):   # add more tracks if necessary
			tfreqn[emptyt] = pfreqt[peaksleft[:emptyt.size]]
			tmagn[emptyt] = pmagt[peaksleft[:emptyt.size]]
			tphasen[emptyt] = pphaset[peaksleft[:emptyt.size]]
			tfreqn = np.append(tfreqn, pfreqt[peaksleft[emptyt.size:]])
			tmagn = np.append(tmagn, pmagt[peaksleft[emptyt.size:]])
			tphasen = np.append(tphasen, pphaset[peaksleft[emptyt.size:]])
	return tfreqn, tmagn, tphasen

def cleaningSineTracks(tfreq, minTrackLength=3):
	"""
	Delete short fragments of a collection of sinusoidal tracks 
	tfreq: frequency of tracks
	minTrackLength: minimum duration of tracks in number of frames
	returns tfreqn: output frequency of tracks
	"""

	if tfreq.shape[1] == 0:                                 # if no tracks return input
		return tfreq
	nFrames = tfreq[:,0].size                               # number of frames
	nTracks = tfreq[0,:].size                               # number of tracks in a frame
	for t in range(nTracks):                                # iterate over all tracks
		trackFreqs = tfreq[:,t]                               # frequencies of one track
		trackBegs = np.nonzero((trackFreqs[:nFrames-1] <= 0)  # begining of track contours
								& (trackFreqs[1:]>0))[0] + 1
		if trackFreqs[0]>0:
			trackBegs = np.insert(trackBegs, 0, 0)
		trackEnds = np.nonzero((trackFreqs[:nFrames-1] > 0)   # end of track contours
								& (trackFreqs[1:] <=0))[0] + 1
		if trackFreqs[nFrames-1]>0:
			trackEnds = np.append(trackEnds, nFrames-1)
		trackLengths = 1 + trackEnds - trackBegs              # lengths of trach contours
		for i,j in zip(trackBegs, trackLengths):              # delete short track contours
			if j <= minTrackLength:
				trackFreqs[i:i+j] = 0
	return tfreq

def sineModel(x, fs, w, N, t):
    """
    Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
    x: input array sound, w: analysis window, N: size of complex spectrum, t: threshold in negative dB 
    returns y: output array sound
    """

    hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
    hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
    Ns = 512                                                # FFT size for synthesis (even)
    H = Ns/4                                                # Hop size used for analysis and synthesis
    hNs = Ns/2                                              # half of synthesis FFT size
    pin = max(hNs, hM1)                                     # init sound pointer in middle of anal window       
    pend = x.size - max(hNs, hM1)                           # last sample to start a frame
    fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
    yw = np.zeros(Ns)                                       # initialize output sound frame
    y = np.zeros(x.size)                                    # initialize output array
    w = w / sum(w)                                          # normalize analysis window
    sw = np.zeros(Ns)                                       # initialize synthesis window
    ow = triang(2*H)                                        # triangular window
    sw[hNs-H:hNs+H] = ow                                    # add triangular window
    bh = blackmanharris(Ns)                                 # blackmanharris window
    bh = bh / sum(bh)                                       # normalized blackmanharris window
    sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window
    while pin<pend:                                         # while input sound pointer is within sound 
    	#-----analysis-----             
        x1 = x[pin-hM1:pin+hM2]                               # select frame
        mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft
        ploc = UF.peakDetection(mX, t)                        # detect locations of peaks
        pmag = mX[ploc]                                       # get the magnitude of the peaks
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
        ipfreq = fs*iploc/float(N)                            # convert peak locations to Hertz
	#-----synthesis-----
        Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)   # generate sines in the spectrum         
        fftbuffer = np.real(ifft(Y))                          # compute inverse FFT
        yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
        yw[hNs-1:] = fftbuffer[:hNs+1] 
        y[pin-hNs:pin+hNs] += sw*yw                           # overlap-add and apply a synthesis window
        pin += H                                              # advance sound pointer

    return y

def sineModelAnal(x, fs, w, N, H, t, maxnSines = 100, minSineDur=.01, freqDevOffset=20, freqDevSlope=0.01):
	"""
	Analysis of a sound using the sinusoidal model with sine tracking
	x: input array sound, w: analysis window, N: size of complex spectrum, H: hop-size, t: threshold in negative dB
	maxnSines: maximum number of sines per frame, minSineDur: minimum duration of sines in seconds
	freqDevOffset: minimum frequency deviation at 0Hz, freqDevSlope: slope increase of minimum frequency deviation
	returns xtfreq, xtmag, xtphase: frequencies, magnitudes and phases of sinusoidal tracks
	"""
	
	if (minSineDur <0):                          # raise error if minSineDur is smaller than 0
		raise ValueError("Minimum duration of sine tracks smaller than 0")
	
	hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
	x = np.append(np.zeros(hM2),x)                          # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hM2))                          # add zeros at the end to analyze last sample
	pin = hM1                                               # initialize sound pointer in middle of analysis window       
	pend = x.size - hM1                                     # last sample to start a frame
	w = w / sum(w)                                          # normalize analysis window
	tfreq = np.array([])
	while pin<pend:                                         # while input sound pointer is within sound            
		x1 = x[pin-hM1:pin+hM2]                               # select frame
		mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft
		ploc = UF.peakDetection(mX, t)                        # detect locations of peaks
		pmag = mX[ploc]                                       # get the magnitude of the peaks
		iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
		ipfreq = fs*iploc/float(N)                            # convert peak locations to Hertz
		# perform sinusoidal tracking by adding peaks to trajectories
		tfreq, tmag, tphase = sineTracking(ipfreq, ipmag, ipphase, tfreq, freqDevOffset, freqDevSlope)
		tfreq = np.resize(tfreq, min(maxnSines, tfreq.size))  # limit number of tracks to maxnSines
		tmag = np.resize(tmag, min(maxnSines, tmag.size))     # limit number of tracks to maxnSines
		tphase = np.resize(tphase, min(maxnSines, tphase.size)) # limit number of tracks to maxnSines
		jtfreq = np.zeros(maxnSines)                          # temporary output array
		jtmag = np.zeros(maxnSines)                           # temporary output array
		jtphase = np.zeros(maxnSines)                         # temporary output array   
		jtfreq[:tfreq.size]=tfreq                             # save track frequencies to temporary array
		jtmag[:tmag.size]=tmag                                # save track magnitudes to temporary array
		jtphase[:tphase.size]=tphase                          # save track magnitudes to temporary array
		if pin == hM1:                                        # if first frame initialize output sine tracks
			xtfreq = jtfreq 
			xtmag = jtmag
			xtphase = jtphase
		else:                                                 # rest of frames append values to sine tracks
			xtfreq = np.vstack((xtfreq, jtfreq))
			xtmag = np.vstack((xtmag, jtmag))
			xtphase = np.vstack((xtphase, jtphase))
		pin += H
	# delete sine tracks shorter than minSineDur
	xtfreq = cleaningSineTracks(xtfreq, round(fs*minSineDur/H))  
	return xtfreq, xtmag, xtphase

def sineModelSynth(tfreq, tmag, tphase, N, H, fs):
	"""
	Synthesis of a sound using the sinusoidal model
	tfreq,tmag,tphase: frequencies, magnitudes and phases of sinusoids
	N: synthesis FFT size, H: hop size, fs: sampling rate
	returns y: output array sound
	"""
	
	hN = N/2                                                # half of FFT size for synthesis
	L = tfreq.shape[0]                                      # number of frames
	pout = 0                                                # initialize output sound pointer         
	ysize = H*(L+3)                                         # output sound size
	y = np.zeros(ysize)                                     # initialize output array
	sw = np.zeros(N)                                        # initialize synthesis window
	ow = triang(2*H)                                        # triangular window
	sw[hN-H:hN+H] = ow                                      # add triangular window
	bh = blackmanharris(N)                                  # blackmanharris window
	bh = bh / sum(bh)                                       # normalized blackmanharris window
	sw[hN-H:hN+H] = sw[hN-H:hN+H]/bh[hN-H:hN+H]             # normalized synthesis window
	lastytfreq = tfreq[0,:]                                 # initialize synthesis frequencies
	ytphase = 2*np.pi*np.random.rand(tfreq[0,:].size)       # initialize synthesis phases 
	for l in range(L):                                      # iterate over all frames
		if (tphase.size > 0):                                 # if no phases generate them
			ytphase = tphase[l,:] 
		else:
			ytphase += (np.pi*(lastytfreq+tfreq[l,:])/fs)*H     # propagate phases
		Y = UF.genSpecSines(tfreq[l,:], tmag[l,:], ytphase, N, fs)  # generate sines in the spectrum         
		lastytfreq = tfreq[l,:]                               # save frequency for phase propagation
		ytphase = ytphase % (2*np.pi)                         # make phase inside 2*pi
		yw = np.real(fftshift(ifft(Y)))                       # compute inverse FFT
		y[pout:pout+N] += sw*yw                               # overlap-add and apply a synthesis window
		pout += H                                             # advance sound pointer
	y = np.delete(y, range(hN))                             # delete half of first window
	y = np.delete(y, range(y.size-hN, y.size))              # delete half of the last window 
	return y

# ASPMA Assignment 10a submission begins here

import sys
import os
from scipy.signal import get_window
	
def sineModelMultiRes(x, fs, Ns, W, M, N, B, T):
    """
    Analysis/synthesis of a sound using the multi-resolution sinusoidal model, without sine tracking
    x:  input array sound,
    fs: sampling frequency, 
    Ns: FFT size for synthesis, 
    W:  array of analysis window types, 
    M:  array of analysis windows sizes, 
    N:  array of sizes of complex spectrums,
    B:  array of frequency bands separators (ascending order of frequency, number of bands == B.size + 1),
    T:  array of peak detection thresholds in negative dB. 
    returns y: output array sound
    """
    
    nResolutions = W.size    
    if (nResolutions != N.size) or (nResolutions != B.size + 1) or (nResolutions != T.size): 
        raise ValueError('Parameters W,N,B,T shall have compatible sizes')    

    H = Ns/4                                                # Hop size used for analysis and synthesis
    hNs = Ns/2                                              # half of synthesis FFT size
    yw = np.zeros(Ns)                                       # initialize output sound frame
    y = np.zeros(x.size)                                    # initialize output array
    sw = np.zeros(Ns)                                       # initialize synthesis window
    ow = triang(2*H)                                        # triangular window
    sw[hNs-H:hNs+H] = ow                                    # add triangular window
    bh = blackmanharris(Ns)                                 # blackmanharris window
    bh = bh / sum(bh)                                       # normalized blackmanharris window
    sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window

    HM1 = map(lambda m: math.floor((m+1)/2),M)              # half analysis windows sizes by rounding
    HM2 = map(lambda m: math.floor( m   /2),M)              # half analysis windows sizes by floor
    maxHM1 = max(HM1)                                       # max half analysis window size by rounding
    pin = max(hNs, maxHM1)                                  # init sound pointers in the middle of largest window       
    pend = x.size - pin                                     # last samples to start a frame
        
    while pin < pend:                                       # while input sound pointer is within sound

        combinedIPFreq = np.array([])
        combinedIPMag  = np.array([])
        combinedIPhase = np.array([])
        windowSizeAttribution = np.array([])
        
        #-----multi-resolution spectrum calculation-----
        for k in range(0,nResolutions):
            windowType = W[k]
            windowSize = M[k]
            w = get_window(windowType,windowSize)                 # normalize analysis window
            w = w / sum(w)
            n = N[k]
            t = T[k]
            hM1 = HM1[k]                                          # half analysis window size by rounding
            hM2 = HM2[k]                                          # half analysis window size by floor
        	#-----analysis-----             
            x1 = x[pin-hM1:pin+hM2]                               # select frame
            mX, pX = DFT.dftAnal(x1, w, n)                        # compute dft
            ploc = UF.peakDetection(mX, t)                        # detect locations of peaks
            iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
            ipfreq = fs*iploc/float(n)                            # convert peak locations to Hertz
            
            if k == 0:    # First frequency range starts from zero 
                f0 = 0.0
            else:
                f0 = B[k-1]
            if k == B.size:    # Last frequency range ends at fs/2
                f1 = fs / 2.0
            else:
                f1 = B[k]
            
            for l in range(0,ipfreq.size):    # Pick the peaks (no pun intended:) inside the assigned frequency band
                f = ipfreq[l]
                if f0 <= f and f < f1:
                    combinedIPFreq = np.append(combinedIPFreq, f)
                    combinedIPMag  = np.append(combinedIPMag , ipmag  [l])
                    combinedIPhase = np.append(combinedIPhase, ipphase[l])
                    windowSizeAttribution = np.append(windowSizeAttribution, windowSize)
            
        
        # Let's smooth out "double-reported" peaks close to the division frequencies of the frequency ranges        
        freqDiffThreshold = (fs*6)/float(n)
        
        smoothedIPFreq = np.array([])
        smoothedIPMag  = np.array([])
        smoothedIPhase = np.array([])
        
        nPeaks = combinedIPFreq.size
        l = 0
        while l < (nPeaks-1):
            f1 = combinedIPFreq[l]
            f2 = combinedIPFreq[l+1]
            m1 = windowSizeAttribution[l]
            m2 = windowSizeAttribution[l+1]
            freqDiff = abs(f1-f2)
            if freqDiff < freqDiffThreshold and m1 != m2:
                #print '!',f1,f2,m1,m2,freqDiff
                smoothedIPFreq = np.append(smoothedIPFreq, (f1+f2)/2.0)
                smoothedIPMag  = np.append(smoothedIPMag , (combinedIPMag [l] + combinedIPMag [l+1])/2.0)
                smoothedIPhase = np.append(smoothedIPhase, (combinedIPhase[l] + combinedIPhase[l+1])/2.0)
                l = l + 2
            else:
                smoothedIPFreq = np.append(smoothedIPFreq, f1)
                smoothedIPMag  = np.append(smoothedIPMag , combinedIPMag [l])
                smoothedIPhase = np.append(smoothedIPhase, combinedIPhase[l])
                l = l + 1
        # Add the last peak        
        smoothedIPFreq = np.append(smoothedIPFreq,combinedIPFreq[nPeaks-1])
        smoothedIPMag  = np.append(smoothedIPMag ,combinedIPMag [nPeaks-1])
        smoothedIPhase = np.append(smoothedIPhase,combinedIPhase[nPeaks-1])

        #-----synthesis-----
        Y = UF.genSpecSines(smoothedIPFreq, smoothedIPMag, smoothedIPhase, Ns, fs)   # generate sines in the spectrum         
        fftbuffer = np.real(ifft(Y))                          # compute inverse FFT
        yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
        yw[hNs-1:] = fftbuffer[:hNs+1] 
        y[pin-hNs:pin+hNs] += sw*yw                           # overlap-add and apply a synthesis window
        pin += H                                              # advance sound pointer

    return y

def calculateAverageQuadraticDifference(x,y):
    s = 0
    # Ignore leading and trailing zeroes in the output
    beg = 0
    end = len(y) - 1
    while y[beg] == 0.0: beg = beg + 1
    while y[end] == 0.0: end = end - 1
    for i in range(beg,end+1):
        d = x[i] - y[i]
        s = s + (d * d)
    sqr = np.sqrt(s)
    aqd = sqr / (end - beg + 1)
    return aqd

def verifySineModelMultiRes(inputFile='../../sounds/orchestra.wav'):
    """
    Checks that the new code returns same result as old one for mono-resolution case 
    inputFile (string) = wav file including the path
    """
    fs, x = UF.wavread(inputFile)               # read input sound
    window = 'hamming'
    m = 2501
    w = get_window(window,m)
    n = 4096
    t = -90.0
    Ns = 512

    y0 = sineModel(x, fs, w, n, t)

    W = np.array([window])
    M = np.array([m])
    N = np.array([n])
    B = np.array([ ])
    T = np.array([t])
    
    y1 = sineModelMultiRes(x, fs, Ns, W, M, N, B, T)

    if y0.size != y1.size: print y0.size,'!=',y1.size
    
    isDifferent = False
    for k in range(0,y0.size):
        if y0[k] != y1[k]:
            isDifferent = True
            print k,y0[k],y1[k],abs(y0[k]-y1[k])
    
    if isDifferent: 
        print 'Results are different in mono-resolution and multi-resolution cases'     
            
        diff = calculateAverageQuadraticDifference(x,y1)
        print 'diff=',diff
    
        #Print out some samples from the input and output, to see how different they are    
        print x [20000:20040]
        print y1[20000:20040]
        
class Best:
    def __init__(self):
        self.diff = sys.float_info.max
        self.Ns = 0
        self.W = []
        self.M = []
        self.N = []
        self.B = []
        self.T = []
    
    def calculateAndUpdate(self,x, fs, Ns, W, M, N, B, T):
        y = sineModelMultiRes(x, fs, Ns, W, M, N, B, T)
        diff = calculateAverageQuadraticDifference(x,y) 
        print diff * 100000,'for W =',W,', M =',M,', N =',N,', B =',B,', T =',T,', Ns =',Ns
        if self.diff > diff:  
            self.diff = diff
            self.Ns = Ns
            self.W = W
            self.M = M
            self.N = N
            self.B = B
            self.T = T
        return y

def exploreSineModelMultiRes(inputFile='../../sounds/orchestra.wav'):
    """
    inputFile (string) = wav file including the path
    """
    fs, x = UF.wavread(inputFile)               # read input sound

    # First, let's check whether the new code returns same result as old one for mono-resolution case    
    
    verifySineModelMultiRes()

   # Let's find optimal parameters in a reasonable range 
  
    windows =['hanning', 'hamming', 'blackman', 'blackmanharris']

    best = Best()
    
    for k in range(5,80,5):
        m = k * 100 + 1                                  # Window size in samples
        for window in windows:                           # Window type
            for t in range(-90,-100,-10):                # Threshold
                for Ns in [512]:                         # size of fft used in synthesis
                    n = 2
                    while n < m: n = n * 2                           # size of fft used in analysis                   
                    for nPower in range(0,3):                        # try out the analysis window closest to window size, and some larger ones 
                        for nAdditionalResolutions in range(0,4):    # try out multi-resolution analysis windows
                            W = np.array([window])
                            M = np.array([m])
                            N = np.array([n])
                            B = np.array([ ])
                            T = np.array([t])

                            log_m = np.log(float(m))
                            log_n = np.log(float(n))
                            log_f = np.log(fs/2.0) 
                            log_step = np.log(2)
                            
                            executeStep = True
                            continueAddingResolutions = True
                            for additionalResolution in range(0,nAdditionalResolutions):
                                if continueAddingResolutions:
                                    scaledM = int(np.exp(log_m - log_step*(additionalResolution+1)))
                                    if scaledM % 2 == 0: scaledM = scaledM + 1
                                    scaledN = int(np.exp(log_n - log_step*(additionalResolution+1)))
                                    if scaledN < scaledM: scaledN = scaledM
                                    appropriateScaledN = 2
                                    while appropriateScaledN < scaledN: appropriateScaledN = appropriateScaledN * 2
                                    frequencyBoundary = np.exp(log_f - (log_step*(nAdditionalResolutions - additionalResolution)))
                                    if scaledM < Ns:
                                        continueAddingResolutions = False
                                        if additionalResolution == 0: executeStep = False
                                    else:
                                        W = np.append(W,window)
                                        M = np.append(M,scaledM)
                                        N = np.append(N,appropriateScaledN)
                                        B = np.append(B,frequencyBoundary)
                                        T = np.append(T,t)
                            if executeStep:
                                best.calculateAndUpdate(x, fs, Ns, W, M, N, B, T)
                        n = n * 2
                        
    print 'FILE:',inputFile
    print 'BEST:','diff =',best.diff,'for W =',best.W,', M =',best.M,', N =',best.N,', B =',best.B,', T =',best.T,', Ns =',best.Ns
    
    y_best = best.calculateAndUpdate(x, fs, best.Ns, best.W, best.M, best.N, best.B, best.T)
    outputFile = inputFile[:-4] + '_optimizedSineModel.wav'
    UF.wavwrite(y_best, fs, outputFile)

def writeExampleFiles():
    """
    A convenience function: writes out example files, some of them with optimal parameters found by exploreSineModelMultiRes()
    """
    inputFile='../../sounds/orchestra.wav'
    fs, x = UF.wavread(inputFile)
    W = np.array(['blackmanharris'])
    M = np.array([1001])
    N = np.array([4096])
    B = np.array([ ])
    T = np.array([-90])
    Ns = 512
    best = Best()
    y = best.calculateAndUpdate(x, fs, Ns, W, M, N, B, T)
    outputFile = inputFile[:-4] + '_optimizedSineModel.wav'
    print '->',outputFile
    UF.wavwrite(y, fs, outputFile)

    inputFile='../../sounds/121061__thirsk__160-link-strings-2-mono.wav'
    fs, x = UF.wavread(inputFile)
    W = np.array(['hamming','hamming','hamming'])
    M = np.array([3001,1501,751])
    N = np.array([16384,8192,4096])
    B = np.array([2756.25,5512.5])
    T = np.array([-90,-90,-90])
    Ns = 512
    best = Best()
    y = best.calculateAndUpdate(x, fs, Ns, W, M, N, B, T)
    outputFile = inputFile[:-4] + '_optimizedSineModel.wav'
    print '->',outputFile
    UF.wavwrite(y, fs, outputFile)

    inputFile='../../sounds/orchestra.wav'
    fs, x = UF.wavread(inputFile)
    W = np.array(['hamming','hamming','hamming'])
    M = np.array([3001,1501,751])
    N = np.array([16384,8192,4096])
    B = np.array([2756.25,5512.5])
    T = np.array([-90,-90,-90])
    Ns = 512
    best = Best()
    y = best.calculateAndUpdate(x, fs, Ns, W, M, N, B, T)
    outputFile = inputFile[:-4] + '_nonOptimizedSineModel.wav'
    print '->',outputFile
    UF.wavwrite(y, fs, outputFile)

    inputFile='../../sounds/121061__thirsk__160-link-strings-2-mono.wav'
    fs, x = UF.wavread(inputFile)
    W = np.array(['blackmanharris'])
    M = np.array([1001])
    N = np.array([4096])
    B = np.array([ ])
    T = np.array([-90])
    Ns = 512
    best = Best()
    y = best.calculateAndUpdate(x, fs, Ns, W, M, N, B, T)
    outputFile = inputFile[:-4] + '_nonOptimizedSineModel.wav'
    print '->',outputFile
    UF.wavwrite(y, fs, outputFile)

if __name__ == "__main__":
    exploreSineModelMultiRes(inputFile='../../sounds/orchestra.wav')
    exploreSineModelMultiRes(inputFile='../../sounds/121061__thirsk__160-link-strings-2-mono.wav')

#    writeExampleFiles()  
