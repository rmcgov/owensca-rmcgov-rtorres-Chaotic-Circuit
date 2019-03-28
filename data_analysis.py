# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anm

# This first piece of code is for op-amp frequency-response analysis

def freq_response_analysis_driver(folderpath, filename="freq_response.npz", dec=8):
    '''
    Perform data analysis on the two-channel voltage response of the op-amp
    (see `frequency_response_test_driver`).
    Arguments:
        folderpath -- path to frequency response results folder
        filename   -- name of response file (.npz) to analyze
        dec        -- decimation at which the response was obtained
    Effects:
        produce a plot (with error bars) of the op-amp gain w/r/t frequency;
        save said plot to disk as vector (.pdf);
        serialize gain-frequency relationship to disk for gain adjustment (.npz)
    '''
    with np.load(folderpath + filename) as scan:
        f = scan['f']
        s1 = scan['s1']
        s2 = scan['s2']
        
        plt.figure()
        plt.plot(f,
                    (np.max(s1, axis=1) - np.min(s1, axis=1)) 
                  / (np.max(s2, axis=1) - np.min(s2, axis=1)))
        plt.show()
        
        def avgpp(a, height):
            '''
            Calculate the average peak-to-peak amplitude of an array, as well
            as its variance
            Arguments:
                a      -- the input array
                height -- the height cutoff to search for peaks with
            Returns:
                [0] average peak-to-peak amplitude for each frequency
                [1] number of peak-to-peak amplitude measurements per frequency
                [2] variance of peak-to-peak amplitude measurements for each frequency
            '''
            s = 125e6 / dec # window length (samples)
            maxes = [sps.find_peaks( a[i,:], height=height, distance=0.5*s/f[i])[0]
                        for i in range(a.shape[0])]
            mins  = [sps.find_peaks(-a[i,:], height=height, distance=0.5*s/f[i])[0]
                        for i in range(a.shape[0])]
            
            max_vals = [a[i,maxes[i]] for i in range(len(maxes))]
            min_vals = [a[i, mins[i]] for i in range(len( mins))]
            
            num_maxes = np.array(list(map(len, maxes)))
            num_mins =  np.array(list(map(len,  mins)))
            num_datapoints = num_maxes + num_mins
            
            avgmax = np.array([np.average(max_vals[i]) for i in range(len(maxes))])
            avgmin = np.array([np.average(min_vals[i]) for i in range(len(mins ))])
            
            varmax = np.array(list(np.sum((max_vals[i] - avgmax[i])**2) for i in range(len(maxes)))) / (num_maxes - 1)
            varmin = np.array(list(np.sum((min_vals[i] - avgmin[i])**2) for i in range(len(mins )))) / (num_mins  - 1)
            
            return avgmax - avgmin, num_datapoints, varmax + varmin
        
        rp_response,   rp_n,   rp_var   = avgpp(s2, 0.7) # tune the height parameters as necessary
        circ_response, circ_n, circ_var = avgpp(s1, 7.0)
        combined_response = circ_response / rp_response
        combined_n = rp_n + circ_n
        combined_stdev = combined_response * np.sqrt((rp_var / rp_response**2) + (circ_var / circ_response**2))
        # NB: this assumes no covariance, which seems unlikely due to feedback
        
#        plt.figure()
#        plt.plot(f, combined_n)
#        plt.plot(f, rp_n)
#        plt.plot(f, circ_n)
#        plt.show()
        
        plt.figure()
        # NB: using chi-squared quantile
        plt.errorbar(f / 1_000, combined_response, yerr=3.84*combined_stdev)
        plt.title("Op Amp Gain: Frequency Dependence")
        plt.xlabel("$f$ (kHz)")
        plt.ylabel("Gain (1)")
        plt.savefig(folderpath + "op-amp-gain.pdf")
        plt.show()
        
        np.savez_compressed(folderpath + "gain_measurement.npz",
                            response=combined_response, n=combined_n, stdev=combined_stdev)
        
        # NB: this is the gain of solely the op amp, but really we do want the STEMlab in the loop for scaling
        
# The remainder of the code is for analysis of RLD voltage traces

def peak_finder(a, width=50):
    '''
    Get (negative) peaks from a voltage trace;
    Assumes peaks are strong enough to be monotone on either side
    Arguments:
        a     -- the array in which to search for peaks
        width -- the width (in samples) of peaks
    Returns:
        an array of peak indices
    '''
#    return sps.find_peaks(-a, width=width, distance=300)
    return sps.argrelmin(a, order=width)[0]

def get_peaks_serial(a, width=50):
    '''
    Get peaks from an array of voltage traces using a serial method
    Arguments:
        a     -- array of voltage traces per input voltage
        width -- the width (in samples) of peaks
    Returns:
        a list containing an array of peak indicies for each input voltage
    '''
    return [peak_finder(a[i,:], width) for i in range(a.shape[0])]

def get_bifurcation_diagram(V, s, width, gain=1, DEBUG=False):
    '''
    Get points for creating a bifurcation diagram, with input voltage on the
    horizontal axis, and peaks on the vertical axis
    Arguments:
        V     -- array of input voltages
        s     -- array of voltage traces per input voltage
        width -- width of peaks (in samples)
        gain  -- an integer or array of gains to multiply input voltage by
    Returns:
        [0] array of input voltages of peaks
        [1] array of magnitudes of peaks
    '''
    peaks = get_peaks_serial(s, width)
    vals  = [s[i,p] for i, p in enumerate(peaks)]
        
    if DEBUG:
        for i in range(0, 200, 200 // 20):
            plt.figure()
            plt.title(V[i])
            plt.plot(s[i,:])
            plt.plot(peaks[i], s[i,peaks[i]], "x")
            plt.show()
    
    x = np.repeat(gain * V, [len(p) for p in peaks])
    y = np.concatenate(vals)
    
    return x, y

def peaks_analysis_driver(folderpath, filename, dec, alpha=0.05):
    '''
    Create a bifurcation diagram from the given voltage scan
    Arguments:
        folderpath -- path to voltage scan results folder
        filename   -- name of voltage scan to analyze (.npz)
        dec        -- the decimation at which the scan was taken
        alpha      -- the transparency of bifurcation diagram points
    Effects:
        produce a bifurcation plot;
        save said plot to disk as vector (.pdf);
        serialize bifurcation diagram points to disk (.npz)
    '''
    with np.load(folderpath + filename) as scan:
        V = scan['V']
        s = scan['s']
        
        width = 400 // dec # tweak this if necessary
        
        # NB: may want to incorporate frequency response for gain
        x, y = get_bifurcation_diagram(V, s, width)
        
        plt.figure()
        plt.plot(x, y, '.', alpha=alpha)
        plt.xlabel("$V_{\mathrm{in}}$ (V)")
        plt.ylabel("$V_{\mathrm{peak}}$ (V)")
        plt.savefig(folderpath + filename.split('.')[0] + "-bifdiag.pdf")
        plt.show()
        
        
        np.savez_compressed(folderpath + filename.split('.')[0] + "-bifdiag.npz",
                            {"x" : x, "y" : y})

def peaks_multiple_analysis_driver(folderpath, filename, dec, alpha=0.05):
    '''
    Create a bifurcation diagram from the given multiple voltage scan
    Arguments:
        folderpath -- path to voltage scan results folder
        filename   -- name of voltage scan to analyze (.npz)
        dec        -- the decimation at which the scan was taken
        alpha      -- the transparency of bifurcation diagram points
    Effects:
        produce a bifurcation plot;
        serialize bifurcation diagram points to disk (.npz)
    '''
    with np.load(folderpath + filename) as scan:
        V = scan['V']
        ss = scan['ss']
        
        width = 400 // dec
        
        xx, yy = [], []
        for j in range(ss.shape[1]):
            x, y = get_bifurcation_diagram(V, ss[:,j,:], width)
            xx.append(x)
            yy.append(y)
            
        xx = np.concatenate(xx)
        yy = np.concatenate(yy)
        
        plt.figure()
        plt.plot(xx, yy, '.', alpha=alpha)
        plt.show()
        
        np.savez_compressed(folderpath + filename.split('.')[0] + "-bifdiag.npz",
                            {"x" : xx, "y" : yy})

def multifrequency_peaks_analysis_driver(folderpath, dec, alpha, forward=True, big=False):
    '''
    Create a bifurcation diagram animation from voltage scans in the given
    folder
    Arguments:
        folderpath -- path to voltage scan results folder
        dec        -- the decimation at which the scans were taken
        alpha      -- the transparency of bifurcation diagram points
        forward    -- if `True`, selects forward scans; 
                      else selects reverse scans
        big        -- if `True`, produces 1080p graphic; 
                      else creates standard 5"x3" graphic
    Effects:
        produce a bifurcation animation;
        save animation to disk (.mp4)
    '''
    fsorted = sorted([(int(f.split('.')[0][1:]), f)
               for f in os.listdir(folderpath) 
               if f.startswith("f" if forward else "r") and f.endswith(".npz")
              ])
    xx, yy = [], []
    for freq, filepath in fsorted:
        with np.load(folderpath + filepath) as scan:
            V, s = scan['V'], scan['s']
            x, y = get_bifurcation_diagram(V, s, 400 // dec)
            xx.append(x)
            yy.append(y)
    
    ff, _ = zip(*fsorted)
        
    fig, ax = plt.subplots(figsize=(20, 11.25) if big else (5, 3), dpi=96)
    ax.set(xlim=(0.,V[-1]), ylim=(np.min(np.concatenate(yy)), 1.),
           xlabel=r"$V_{\mathrm{in}}$ (V)",
           ylabel=r"$V_{\mathrm{out}}$ (V)",
           title=f"$f$={ff[0]}Hz")
    line = ax.plot(xx[0], yy[0], '.', alpha=alpha)[0]
    
    def animate(i):
        line.set_data(xx[i], yy[i])
        ax.set_title(f"$f$={ff[i]}Hz")
    
    anim = anm.FuncAnimation(fig, animate, interval=1000, frames=len(ff))
    anim.save(folderpath + "animation.mp4")
#    anim.save(folderpath + "animation_small.gif", 
#              writer=anm.FFMpegWriter(codec="gif", fps=1))
    
    plt.draw()
    plt.show()

def poincare_plot_driver(folderpath, filename, dec, alpha=0.2, big=False):
    '''
    Create a 2D Poincaré recurrence plot for the peak heights of the given 
    voltage scan, animated based on input voltage
    Arguments:
        folderpath -- path to voltage scan results folder
        filename   -- name of voltage scan to analyze (.npz)
        dec        -- the decimation at which the scan was taken
        alpha      -- the transparency of bifurcation diagram points
        big        -- if `True`, produces 1080p graphic; 
                      else creates standard 5"x3" graphic
    Effects:
        produce a Poincaré recurrence animation;
        save said animation to disk (.mp4)
    '''
    with np.load(folderpath + filename) as scan:
        V, s = scan['V'], scan['s']
        peaks = get_peaks_serial(s, 400 // dec)
        peak_vals = [s[i,p] for i, p in enumerate(peaks)]
        
        infreq = int(filename.split('.')[0].split('-')[0][1:])

#        for i in range(0, len(peak_vals), len(peak_vals) // 40):
#            plt.figure()
#            plt.plot(peak_vals[i][:-1], peak_vals[i][1:], '.', alpha=alpha)
#            plt.show()
        
        fig, ax = plt.subplots(figsize=(20, 11.25) if big else (5, 3), dpi=96)
        poin, = ax.plot(peak_vals[0][:-1], peak_vals[0][1:], '.', alpha=alpha)
        titlepre = "Poincaré Recurrence: " \
                   + f"$f={infreq}$ Hz, "  \
                   + "$V_{\mathrm{in}}=$"
        ax.set(xlim=(np.min(s), np.max(s)),
               ylim=(np.min(s), np.max(s)),
               xlabel="$V_{\mathrm{out}}(n)$ (V)",
               ylabel="$V_{\mathrm{out}}(n+1)$ (V)",
               title=titlepre+f"{V[0]:.4f} V"
               )
        
        def animate(i):
            ax.set(title=titlepre+f"{V[i]:.4f} V")
            poin.set(xdata=peak_vals[i][:-1],
                     ydata=peak_vals[i][1:],
                    )
        
        anim = anm.FuncAnimation(fig, animate, interval=1000//15, frames=len(peaks))
        anim.save(folderpath + filename.split(".")[0] + "-poincare.mp4")
        
        plt.draw()
        plt.show()

def poincare_plot_3d_driver(folderpath, filename, dec, alpha=0.05, big=False):
    '''
    Creates a 3D visualization of 2D Poincaré recurrence plots, stacked along 
    input voltage, animated to rotate
    Arguments:
        folderpath -- path to voltage scan results folder
        filename   -- name of voltage scan to analyze (.npz)
        dec        -- the decimation at which the scan was taken
        alpha      -- the transparency of bifurcation diagram points
        big        -- if `True`, produces 1080p graphic; 
                      else creates standard 5"x3" graphic
    Effects:
        produce a Poincaré recurrence visualization animation;
        save said animation to disk (.mp4)
    '''
    with np.load(folderpath + filename) as scan:
        V, s = scan['V'], scan['s']
        peaks = get_peaks_serial(s, 400 // dec)
        peak_vals = [s[i,p] for i, p in enumerate(peaks)]
        
        infreq = int(filename.split('.')[0].split('-')[0][1:])
        
        VV = [[V[i]] * (len(p) - 1) for i, p in enumerate(peaks)]
        VV = [v for vv in VV for v in vv]
        xx = [p[:-1] for p in peak_vals]
        xx = [p for pp in xx for p in pp]
        yy = [p[1:] for p in peak_vals]
        yy = [p for pp in yy for p in pp]
        
        fig = plt.figure(figsize=(20, 11.25) if big else (5, 3), dpi=96)
        fig.suptitle(f"Poincaré Recurrence: $f={infreq}$ Hz")
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xx, yy, VV, '.', alpha=alpha)
        ax.set(xlabel="$V_{\mathrm{out}}(n)$ (V)",
               ylabel="$V_{\mathrm{out}}(n+1)$ (V)",
               zlabel="$V_{\mathrm{in}} (V)$"
               )
        
        angles = np.linspace(0, 360, 15*30)
        
        def animate(i):
            ax.view_init(azim=angles[i])
            return fig,
        
        anim = anm.FuncAnimation(fig, animate, frames=len(angles), interval=1000//15, blit=True)
        anim.save(folderpath + filename.split(".")[0] + "-poincare-bifdiag.mp4")
        

def forward_backward_asymmetry_analysis_driver(fwdfolderpath, fwdfilename, 
                                               revfilepath = None, revfilename = None,
                                               dec = 0, alpha = None):
    '''
    Create two overlaid bifurcation diagrams, one for a forward voltage scan,
    and one for a reverse voltage scan, to directly compare them
    Arguments:
        fwdfolderpath -- path to forward voltage scan results folder
        fwdfilename   -- name of forward voltage scan to analyze (.npz)
        revfilepath   -- path to reverse voltage scan results folder
                         (optional; if not provided inferred from forward)
        revfilename   -- name of reverse voltage scan to analyze
                         (optional; if not provided inferred from forward)
        dec           -- the decimation at which the scans were taken
        alpha         -- the transparency of bifurcation diagram points
    Effects:
        produce the overlaid bifurcation diagram
    '''
    if revfilepath == None:
        revfilepath = fwdfolderpath
    if revfilename == None:
        revfilename = "r" + fwdfilename[1:]
    if alpha == None:
        alpha = 0.01/np.log(dec)
    
    with np.load(fwdfolderpath + fwdfilename) as fwd, \
         np.load(revfilepath + revfilename) as rev:
        Vf, sf = fwd['V'], fwd['s']
        Vr, sr = rev['V'], rev['s']
        
        width = 400 // dec        
        xf, yf = get_bifurcation_diagram(Vf, sf, width)
        xr, yr = get_bifurcation_diagram(Vr, sr, width)
        
        plt.figure()
        plt.plot(xf, yf, '+', alpha=alpha, label="forward")
        plt.plot(xr, yr, 'x', alpha=alpha, label="reverse")
        leg = plt.legend()
        for lh in leg.legendHandles: 
            lh._legmarker.set_alpha(1)
        plt.show()

def voltage_scan_spectrum_driver(folderpath, filename, dec, colorbar=True):
    '''
    Create a frequency waterfall visualization of the spectrum of voltage 
    traces as a function of input voltage
    NB: unlike traditional waterfalls, these flow from left to right to match
    the corresponding bifurcation diagrams
    Arguments:
        folderpath -- path to voltage scan results folder
        filename   -- name of voltage scan to analyze (.npz)
        dec        -- the decimation at which the scan was taken
        colorbar   -- if `True`, includes a colorbar scale
    Effects:
        produce the waterfall visualization;
        save said visualization to disk (.png)
    '''
    with np.load(folderpath + filename) as scan:
        V, s = scan['V'], scan['s']
        infreq = int(filename.split('.')[0].split('-')[0][1:])
        
        d = dec / (125e6) # sample length (s)
        t = np.arange(0, 2**14*d, d)
        
        for window in [
#                       sps.boxcar, 
#                       sps.triang, 
#                       sps.blackman, 
#                       sps.hamming, 
                       sps.hann, 
#                       sps.bartlett, 
#                       sps.flattop, 
#                       sps.parzen, 
#                       sps.bohman, 
#                       sps.blackmanharris, 
#                       sps.nuttall, 
#                       sps.barthann
                       ]:
            f, _, spec = sps.spectrogram(np.ravel(s), fs=(1 / d), nfft=2**14, noverlap=0, window=window(2**14))
            fig, ax = plt.subplots()
            pc = ax.pcolormesh(V, f / 1_000, spec, norm=colors.LogNorm())
            if colorbar:
                cb = plt.colorbar(pc)
                cb.set_label("$I$ (W/Hz)")
            ax.set_xlabel("$V_{\mathrm{in}}$ (V)")
            ax.set_ylabel("$f$ (kHz)")
            ax.set_ylim(0, 2 * (infreq / 1_000))
            # NB: do NOT serialize to .pdf, it doesn't play nice
            plt.savefig(folderpath + filename.split('.')[0] + "-spectrum.png")
            plt.show()

def multifrequency_voltage_scan_spectrum_driver(folderpath, dec, forward=True, colorbar=True, big=False):
    '''
    Create a frequency waterfall visualization of the spectra of voltage traces
    as a function of input voltage, animated based on input frequency
    NB: unlike traditional waterfalls, these flow from left to right to match
    the corresponding bifurcation diagrams
    Arguments:
        folderpath -- path to voltage scan results folder
        dec        -- the decimation at which the scans were taken
        forward    -- if `True`, selects forward scans; 
                      else selects reverse scans
        colorbar   -- if `True`, includes a colorbar scale
        big        -- if `True`, produces 1080p graphic; 
                      else creates standard 5"x3" graphic
    Effects:
        produce the waterfall animation;
        save said animation to disk (.mp4)
    '''
    fsorted = sorted([(int(f.split('.')[0].split('-')[0][1:]), f)
               for f in os.listdir(folderpath) 
               if f.startswith("f" if forward else "r") and f.endswith(".npz")
              ])
    
    SS = []
    for infreq, filename in fsorted:
        with np.load(folderpath + filename) as scan:
            V, s = scan['V'], scan['s']
            d = dec / (125e6) # sample length (s)
            f, _, S = sps.spectrogram(np.ravel(s), fs=(1 / d), nfft=2**14, noverlap=0, window=sps.hann(2**14))
            SS.append(S)
            
    infreqs, _ = zip(*fsorted)
    
    fig, ax = plt.subplots(figsize=(20, 11.25) if big else (5, 3), dpi=96)
    pc = ax.pcolormesh(V, f / 1_000, SS[0], norm=colors.LogNorm())
    if colorbar:
        cb = plt.colorbar(pc)
        cb.set_label=("$I$ (W/Hz)")
    ax.set(title=f"$f$={infreqs[0]}Hz",
           xlabel="$V_{\mathrm{in}}$ (V)",
           ylabel="$f$ (kHz)",
           ylim=(0, 2 * (infreqs[0] / 1_000))
          )
    
    def animate(i):
        pc = ax.pcolormesh(V, f / 1_000, SS[i], norm=colors.LogNorm())
        ax.set(title=f"$f$={infreqs[i]}Hz",
               ylim=(0, 2 * (infreqs[i] / 1_000)),
              )
        return pc,
    
    anim = anm.FuncAnimation(fig, animate, interval=1000, frames=len(infreqs))
    anim.save(folderpath + "spec-animation.mp4")
    
    plt.draw()
    plt.show()
 
def voltage_scan_cepstrum_driver(folderpath, filename, dec):
    '''
    Create a quefrency waterfall visualization
    Arguments:
        folderpath -- path to voltage scan results folder
        filename   -- name of voltage scan to analyze (.npz)
        dec        -- the decimation at which the scan was taken
    Effects:
        produce the waterfall visualization
    '''
    with np.load(folderpath + filename) as scan:
        V, s = scan['V'], scan['s']
        infreq = int(filename.split('.')[0].split('-')[0][1:])
        d = dec / 125e6 # sample length (s)
        t = np.arange(0, 2**14*d, d)
        
        # taking ihfft of abs of rfft w/ triang window seems to give best results
        spectrum       = np.abs(np.fft.rfft(s * sps.triang(2**14), axis=1))
        spectrum_freqs = np.fft.rfftfreq(s.shape[1], d)
        cepstrum       = np.abs(np.fft.ihfft(np.log(spectrum), axis=1)) # irfft?
        cepstrum_quefs = np.arange(0, 2**14)
        
#        for i in range(0, len(V), len(V)//5):
#            plt.figure()
#            plt.semilogy(spectrum_freqs, spectrum[i,:])
#            plt.show()
#            plt.figure()
#            plt.semilogy(cepstrum[i,:])
#            plt.xlim(0, 2 / d / infreq)
#            plt.show()
        max_quef = int(4//(d*infreq))
        fig, ax1 = plt.subplots()
        ax1.pcolormesh(V, cepstrum_quefs[:max_quef], np.transpose(cepstrum[:,:max_quef]), norm=colors.LogNorm())
        ax2 = ax1.twinx()
        ax2.plot(*get_bifurcation_diagram(V, s, 400 // dec), '.', alpha=0.01/np.log(dec), color='r')
        ax2.set_xlim(V[0], V[-1])
        plt.show()
  
def multifrequency_voltage_scan_dual_animation_driver(folderpath, dec, alpha, forward=True, colorbar=True, big=False):
    '''
    Create a visualization of both a bifurcation diagram and frequency 
    waterfall spectrum, animated based on input frequeny, from the given
    voltage scans
    Arguments:
        folderpath -- path to voltage scan results folder
        dec        -- the decimation at which the scans were taken
        forward    -- if `True`, selects forward scans; 
                      else selects reverse scans
        colorbar   -- if `True`, includes a colorbar scale
        big        -- if `True`, produces 1080p graphic; 
                      else creates standard 5"x3" graphic
    Effects:
        produce the animation;
        save the animation to disk
    '''
    fsorted = sorted([(int(f.split('.')[0].split('-')[0][1:]), f)
               for f in os.listdir(folderpath) 
               if f.startswith("f" if forward else "r") and f.endswith(".npz")
              ])
    
    width = 400 // dec
    d = dec / (125e6) # sample length (s)
    xx, yy, SS = [], [], []
    for infreq, filepath in fsorted:
        with np.load(folderpath + filepath) as scan:
            V, s = scan['V'], scan['s']
            x, y = get_bifurcation_diagram(V, s, width)
            f, _, S = sps.spectrogram(np.ravel(s), fs=(1 / d), nfft=2**14, noverlap=0, window=sps.hann(2**14))
            SS.append(S)
            xx.append(x)
            yy.append(y)
    
    infreqs, _ = zip(*fsorted)
    
    fig = plt.figure(figsize=(20, 11.25) if big else (5, 3), dpi=96)
    fig.suptitle(f"$f$={infreqs[0]}Hz")
    
    ax1 = fig.add_subplot(211)
    ax1.set(title="Bifurcation diagram",
            xlim=(0.,V[-1]), ylim=(np.min(np.concatenate(yy)), 1.),
            xlabel=r"$V_{\mathrm{in}}$ (V)",
            ylabel=r"$V_{\mathrm{out}}$ (V)"
           )
    bifdiag = ax1.plot(xx[0], yy[0], '.', alpha=alpha)[0]
    
    ax2 = fig.add_subplot(212)
    spectrum = ax2.pcolormesh(V, f / 1_000, SS[0], norm=colors.LogNorm())
    if colorbar:
        cb = plt.colorbar(spectrum)
        cb.set_label=("$I$ (W/Hz)")
    ax2.set(title="Spectrum",
            xlabel="$V_{\mathrm{in}}$ (V)",
            ylabel="$f$ (kHz)",
            ylim=(0, 2 * (infreqs[0] / 1_000))
           )
    
    def animate(i):
        fig.suptitle(f"$f$={infreqs[i]}Hz")
        bifdiag.set_data(xx[i], yy[i])
        spectrum = ax2.pcolormesh(V, f / 1_000, SS[i], norm=colors.LogNorm())
        ax2.set_ylim(0, 2 * (infreqs[i] / 1_000))
        return bifdiag, spectrum
    
    anim = anm.FuncAnimation(fig, animate, interval=1000, frames=len(infreqs))
    anim.save(folderpath + "dual-animation.mp4")
    
    plt.close()
    
def voltage_scan_animation_driver(folderpath, filename, dec, big=False, subtitles=False):
    '''
    Create a deluxe visualization based on the given voltage scan:
        - Voltage trace animated based on input voltage
        - Bifurcation diagram
        - Frequency spectrum animated based on input voltage
        - Frequency waterfall diagram
    Arguments:
        folderpath -- path to voltage scan results folder
        filename   -- name of voltage scan to analyze (.npz)
        dec        -- the decimation at which the scan was taken
        big        -- if `True`, produces 1080p graphic; 
                      else creates standard 5"x3" graphic
        subtitles  -- if `True`, put descriptive titles on the subfigures
    Effects:
        produce the animation;
        save said animation to disk (.mp4)
    '''
    with np.load(folderpath + filename) as scan:
        V, s = scan['V'], scan['s']
        infreq = int(filename.split('.')[0][1:])
        
        d = dec / (125e6) # sample length (s)
        t = np.arange(0, 2**14*d, d)
        
        ffts = np.fft.rfft(s, axis=1)
        fft_freqs = np.fft.rfftfreq(s.shape[1], d)
        
        bif_x, bif_y = get_bifurcation_diagram(V, s, 400//dec)
        
        f, _, S = sps.spectrogram(np.ravel(s), fs=(1 / d), nfft=2**14, noverlap=0, window=sps.hann(2**14))
        
#        for i in range(0, len(V), len(V)//20):
#            trace = s[i,:]
##            plt.figure()
##            plt.plot(t, trace)
##            plt.show()
#            plt.figure()
#            plt.title(V[i])
#            plt.plot(np.fft.rfftfreq(len(trace), d), 
#                     np.abs(np.fft.rfft(trace)))
#            plt.xlim(0, infreq * 5)
#            plt.show()
        
        fig = plt.figure(figsize=(20, 11.25) if big else (5, 3), dpi=96)
        fig.suptitle(f"$f$={infreq}Hz, $V$={V[0]}")
        ax1 = fig.add_subplot(221)
        ax1.set(xlim=(0.001, 0.002) if dec==64 else None,
                ylim=(np.min(s), np.max(s)),
                xlabel="$t$ (s)",
                ylabel="$V$ (V)",
                title="Trace" if subtitles else ""
                )
        ax2 = fig.add_subplot(222)
        ax2.set(xlim=(V[0], V[-1]),
                xlabel="$V_{\mathrm{in}}$ (V)",
                ylabel="$V_{\mathrm{out}}$ (V)",
                title="Bifurcaion Diagram" if subtitles else ""
                )
        ax3 = fig.add_subplot(223)
        ax3.set(xlim=(0, infreq*2),
                ylim=(np.min(np.abs(ffts)), np.max(np.abs(ffts))),
                yticks=[],
                yticklabels=[],
                xlabel="$f$ (Hz)",
                title="Spectrum" if subtitles else ""
                )
        ax4 = fig.add_subplot(224)
        ax4.set(xlabel="$V_{\mathrm{in}}$ (V)",
                ylabel="$f$ (kHz)",
                ylim=(0, 2 * (infreq / 1_000)),
                title="Waterfall" if subtitles else ""
                )
        trace = ax1.plot(t, s[0,:])[0]
        bif   = ax2.plot(bif_x, bif_y, '.', alpha=1/dec)
        line1 = ax2.axvline(V[0], color='r')
        fft   = ax3.plot(fft_freqs, np.abs(ffts[0,:]))[0]
        wfall = ax4.pcolormesh(V, f / 1_000, S, norm=colors.LogNorm())
        line2 = ax4.axvline(V[0], color='r', alpha=0.3)
        
        def animate(i):
            fig.suptitle(f"$f$={infreq}Hz, $V$={V[i]}")
            trace.set_ydata(s[i,:])
            line1.set_xdata([V[i]] * 2)
            fft.set_ydata(np.abs(ffts[i,:]))
            line2.set_xdata([V[i]] * 2)
        
        anim = anm.FuncAnimation(fig, animate, interval=500, frames=len(V))
        anim.save(folderpath + filename.split(".")[0] + "-anim.mp4")
#        anim.save(folderpath + filename.split(".")[0] + "-anim-small.gif",
#                  anm.FFMpegWriter(codec="gif", fps=1)
#                  )
        
        plt.draw()
        plt.show()

if __name__ == "__main__":
    # call the analyses you want to run here
    pass
    
    
