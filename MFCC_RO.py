'''MFCC.py
Calculation of MFCC coefficients from frequency-domain data
Adapted from the Vampy example plugin "PyMFCC" by Gyorgy Fazekas
http://code.soundsoftware.ac.uk/projects/vampy/repository/entry/Example%20VamPy%20plugins/PyMFCC.py
Centre for Digital Music, Queen Mary University of London.
Copyright (C) 2009 Gyorgy Fazekas, QMUL.
'''

import sys,numpy
from numpy import abs,log,exp,floor,sum,sqrt,cos,hstack
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import stride_tricks
#from scipy.io import wavfile
import scipy.io as sio

class melScaling(object):

    def __init__(self,sampleRate,inputSize,numBands,minHz = 0,maxHz = None):
        '''Initialise frequency warping and DCT matrix. 
        Parameters:
        sampleRate: audio sample rate
        inputSize: length of magnitude spectrum (half of FFT size assumed)
        numBands: number of mel Bands (MFCCs)
        minHz: lower bound of warping  (default = DC)
        maxHz: higher bound of warping (default = Nyquist frequency)
        '''
        self.sampleRate = sampleRate
        self.NqHz = sampleRate / 2.0
        self.minHz = minHz
        if maxHz is None : maxHz = self.NqHz
        self.maxHz = maxHz
        self.inputSize = inputSize
        self.numBands = numBands
        self.valid = False
        self.updated = False
        
    def update(self): 
        # make sure this will run only once 
        # if called from a vamp process
        if self.updated: return self.valid
        self.updated = True
        self.valid = False
        #print('Updating parameters and recalculating filters: ')
        #print('Nyquist: ',self.NqHz)
        
        if self.maxHz > self.NqHz : 
            raise Exception('Maximum frequency must be smaller than the Nyquist frequency')
        
        self.maxMel = 1000*log(1+self.maxHz/700.0)/log(1+1000.0/700.0)
        self.minMel = 1000*log(1+self.minHz/700.0)/log(1+1000.0/700.0)
        print('minHz:%s\nmaxHz:%s\nminMel:%s\nmaxMel:%s\n' \
        %(self.minHz,self.maxHz,self.minMel,self.maxMel))
        self.filterMatrix = self.getFilterMatrix(self.inputSize,self.numBands)
        self.DCTMatrix = self.getDCTMatrix(self.numBands)
        self.valid = True
        return self.valid
                
    def getFilterCentres(self,inputSize,numBands):
        '''Calculate Mel filter centres around FFT bins.
        This function calculates two extra bands at the edges for
        finding the starting and end point of the first and last 
        actual filters.'''
        centresMel = numpy.array(range(numBands+2)) * (self.maxMel-self.minMel)/(numBands+1) + self.minMel
        centresBin = numpy.floor(0.5 + 700.0*inputSize*(exp(centresMel*log(1+1000.0/700.0)/1000.0)-1)/self.NqHz)
        return numpy.array(centresBin,int)
        
    def getFilterMatrix(self,inputSize,numBands):
        '''Compose the Mel scaling matrix.'''
        filterMatrix = numpy.zeros((numBands,inputSize))
        self.filterCentres = self.getFilterCentres(inputSize,numBands)
        for i in range(numBands) :
            start,centre,end = self.filterCentres[i:i+3]
            self.setFilter(filterMatrix[i],start,centre,end)
        return filterMatrix.transpose()

    def setFilter(self,filt,filterStart,filterCentre,filterEnd):
        '''Calculate a single Mel filter.'''
        k1 = numpy.float32(filterCentre-filterStart)
        k2 = numpy.float32(filterEnd-filterCentre)
        up = (numpy.array(range(filterStart,filterCentre))-filterStart)/k1
        dn = (filterEnd-numpy.array(range(filterCentre,filterEnd)))/k2
        filt[filterStart:filterCentre] = up
        filt[filterCentre:filterEnd] = dn

    def warpSpectrum(self,magnitudeSpectrum):
        '''Compute the Mel scaled spectrum.'''
        return numpy.dot(magnitudeSpectrum,self.filterMatrix)
        
    def getDCTMatrix(self,size):
        '''Calculate the square DCT transform matrix. Results are 
        equivalent to Matlab dctmtx(n) with 64 bit precision.'''
        DCTmx = numpy.array(range(size),numpy.float64).repeat(size).reshape(size,size)
        DCTmxT = numpy.pi * (DCTmx.transpose()+0.5) / size
        DCTmxT = (1.0/sqrt( size / 2.0)) * cos(DCTmx * DCTmxT)
        DCTmxT[0] = DCTmxT[0] * (sqrt(2.0)/2.0)
        return DCTmxT
        
    def dct(self,data_matrix):
        '''Compute DCT of input matrix.'''
        return numpy.dot(self.DCTMatrix,data_matrix)
        
    def getMFCCs(self,warpedSpectrum,cn=True):
        '''Compute MFCC coefficients from Mel warped magnitude spectrum.'''
        mfccs=self.dct(numpy.log(numpy.clip(warpedSpectrum, 1e-9, numpy.inf)))
        if cn is False : mfccs[0] = 0.0
        return mfccs
    
def surf_plt(data):
    nx, ny = data.shape
    x = range(nx)
    y = range(ny)
    hf = plt.figure() 
    ha = hf.add_subplot(111, projection = '3d')
     
    X, Y = np.meshgrid(x, y)
    ha.plot_surface(X, Y, data.T)
    
    plt.show()

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames) 

def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs

def nornalizar(data):
    x , y = data.shape
    data = data.flatten()
    media = np.mean(data)
    var = np.var(data) + 10
    
    data = np.maximum(0, data - media)/np.sqrt(var)
    return np.reshape(data, [x, y])

def normalise_and_whiten(data, retain=0.99, bias=1e-8, use_selfnormn=True, min_ndims=1):
    "Use this to prepare a training set before running through OSKMeans"
    mean = np.mean(data, 0)
    normdata = data - mean

    if use_selfnormn:
        for i in range(normdata.shape[0]):
            normdata[i] -= np.mean(normdata[i])

    # this snippet is based on an example by Sander Dieleman
    cov = np.dot(normdata.T, normdata) / normdata.shape[0]
    eigs, eigv = np.linalg.eigh(cov) # docs say the eigenvalues are NOT NECESSARILY ORDERED, but this seems to be the case in practice...
    print "  computing number of components to retain %.2f of the variance..." % retain
    normed_eigs = eigs[::-1] / np.sum(eigs) # maximal value first
    eigs_sum = np.cumsum(normed_eigs)
    num_components = max(min_ndims, np.argmax(eigs_sum > retain)) # argmax selects the first index where eigs_sum > retain is true
    print "  number of components to retain: %d of %d" % (num_components, len(eigs))
    P = eigv.astype('float32') * np.sqrt(1.0/(eigs + bias)) # PCA whitening
    P = P[:, -num_components:] # truncate transformation matrix
     
    whitedata = np.dot(normdata, P)
    invproj = np.linalg.pinv(P)
    return ({'centre': mean, 'proj': P, 'ncomponents': num_components, 'invproj': invproj, 'use_selfnormn': use_selfnormn}, whitedata)

def prepare_data(data, norminfo):
    "Typically used for new data; you use normalise_and_whiten() on your training data, then this method projects a new set of data rows in the same way"
    normdata = data - norminfo['centre']
    try:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        if norminfo['use_selfnormn']:
            normdata -= np.mean(normdata, 1).reshape(-1,1)
        return np.dot(normdata, norminfo['proj'])
    except:
        print("Matrix shapes were: data %s, norminfo['proj'] %s, np.mean(normdata, 1) %s" % (np.shape(normdata), np.shape(norminfo['proj']), np.shape(np.mean(normdata, 1))))
        raise

def prepare_a_datum(datum, norminfo):
    "Typically used for new data; you use normalise_and_whiten() on your training data, then this method projects a single test datum in the same way"
    return prepare_data(datum.reshape(1, -1), norminfo).flatten()

if __name__ == "__main__":
    
    
    contador = 0
    fs = 44100
    frame_len = int(512)
    mfccMaker = melScaling(sampleRate = fs, inputSize = 49, numBands = 40, minHz = 500, maxHz = None)
    mfccMaker.update()
    
    file_name = list('/Users/yasushishibe/Documents/data_clef/t1.mat')
    
    save_tx = list("/Users/yasushishibe/Documents/workspace/TF/src/data3.0/save/whitedata_train_x_1.npy")
    save_ty = list("/Users/yasushishibe/Documents/workspace/TF/src/data3.0/save/whitedata_train_y_1.npy")
    
    for i in range(1,9686):    
        file_name[41]= str(i)
        file_data = sio.loadmat("".join(file_name))
        train_x = file_data['train_x']
        train_y = file_data['train_y']
        
        for l in range(train_x.shape[1]):
            
            complexSpectrum = train_x[0,l].flatten()
            espectrostft = stft(complexSpectrum, frameSize=frame_len, overlapFac=0.663, window=np.hanning)
            espectrostft = logscale_spec(espectrostft, sr=fs, factor=20.)
            magnitudeSpectrum = np.abs(espectrostft[0])
            melSpectrum = mfccMaker.warpSpectrum(magnitudeSpectrum)
            melCepstrum = mfccMaker.getMFCCs(melSpectrum.T,cn=True)
            norm = nornalizar(melSpectrum)
            data = np.maximum(0, norm - np.mean(norm, 0))
            
            dif1 = np.diff(data, n=1, axis=0)
            dif2 = np.diff(data, n=2, axis=0)
            diff_len = dif2.shape[0]
            data = np.append(np.append(data[:diff_len,:], dif1[:diff_len,:], axis= 1), dif2, axis=1)
                
            
            if i == 1:
                norminfo, whitedata = normalise_and_whiten(data, retain=0.99, bias=1e-8, use_selfnormn=True, min_ndims=16)  #codigo para el blanqueamiento de los datos
                    
            else:
                whitedata = np.zeros([data.shape[0], norminfo['ncomponents']])
                for p in range(data.shape[0]):
                    whitedata[p,:] = prepare_a_datum(data[p,:], norminfo)
            
              
            
            contador = contador + 1
            save_tx[78] = str(contador)
            save_ty[78] = str(contador)
            np.save("".join(save_tx), whitedata)
            np.save("".join(save_ty), [train_y[0,l], i])
            
            
        print "Iteracion numero: "+repr(i)
        
    
    
