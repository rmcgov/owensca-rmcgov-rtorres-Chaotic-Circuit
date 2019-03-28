# -*- coding: utf-8 -*-
import os
import visa
#import struct
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def get_resource_manager():
    '''
    Acquire the pyVISA resource manager.
    Returns:
        the resource manager
    '''
    return visa.ResourceManager()

def get_rp(rm, ip, port=5000):
    '''
    Acquire the STEMlab device.
    Arguments:
        rm   -- the pyVISA resource manager
        ip   -- the ip of the STEMlab device
        port -- the port to address the STEMlab device
    Throws:
        AssertionError -- invalid device name on acquisition
    Returns:
        the STEMlab device
    '''
    rp = rm.open_resource(f"TCPIP::{ip}::{port}::SOCKET", 
                          read_termination = "\r\n")
    assert(rp.query("*IDN?") == "your redpitaya's name here")
    return rp

def config_rp_fg(osc, sour=2):
    '''
    Configure the function-generator (output) of the STEMlab device.
    Arguments:
        osc  -- the STEMlab device
        sour -- the channel to use for output voltage
    '''
    osc.write(f"OUTPUT{sour}:STATE ON")
    osc.write(f"SOUR{sour}:FUNC SINE")

def config_rp_osc(osc, fmt, dec=8):
    '''
    Configure the oscilloscope (input) of the STEMlab device.
    Arguments:
        osc -- the STEMlab device
        fmt -- the data format; ("ASCII" | "DEC")
        dec -- the decimation
    '''
    osc.write(f"ACQ:DEC {dec}")
    osc.write("ACQ:SOUR1:GAIN HV")
    osc.write("ACQ:SOUR2:GAIN HV")
    # NB: this is a magic number of samples until the voltage "kicks in",
    #     paradoxically scaling up with decimation?
    osc.write("ACQ:TRIG:DLY 8500")
    if fmt == "ASCII":
        osc.write("ACQ:DATA:FORMAT ASCII")
        osc.write("ACQ:DATA:UNITS VOLTS")
    elif fmt == "BIN":
        osc.write("ACQ:DATA:FORMAT BIN")
        osc.write("ACQ:DATA:UNITS RAW")
    else:
        raise ValueError(f"Invalid format {fmt}")

def acquire_ascii(osc, chan=[1]):
    '''
    Acquire ASCII data from an oscilloscope-like device.
    Arguments:
        osc  -- the device
        chan -- an iterable of channels to acquire from
    Returns:
        a tuple of numpy arrays, one per channel
    '''
    osc.write("ACQ:START")
    osc.write("ACQ:TRIG NOW")
    
    osc.clear()
    while True:
        if osc.query("ACQ:TRIG:STAT?") == "TD":
            break
    
    osc.write("ACQ:STOP")
    return tuple(np.fromstring(osc.query(f"ACQ:SOUR{i}:DATA?").strip("{}"), sep=",")
            for i in chan)
 
def acquire_ascii_2channels(osc):
    '''
    Acquire ASCII data from the first 2 channels of an oscilloscope-like
    device.
    Arguments:
        osc -- the device
    Returns:
        a tuple of numpy arrays, one per channel
    '''
    osc.write("ACQ:START")
    osc.write("ACQ:TRIG NOW")
    
    osc.clear()
    while True:
        if osc.query("ACQ:TRIG:STAT?") == "TD":
            break
    
    osc.write("ACQ:STOP")
    return (np.fromstring(osc.query(f"ACQ:SOUR1:DATA?").strip("{}"), sep=","),
            np.fromstring(osc.query("ACQ:SOUR2:DATA?").strip("{}"), sep=","))

def acquire_binary(osc, chan=[1]):
    '''
    Acquire binary data from an oscilloscope-like device.
    Arguments:
        osc  -- the device
        chan -- an iterable of channels to acquire from
    Returns:
        a tuple of numpy arrays, one per channel
    '''
    osc.write("ACQ:START")
    osc.write("ACQ:TRIG NOW")
    
    osc.clear()
    while True:
        if osc.query("ACQ:TRIG:STAT?") == "TD":
            break
    
    osc.write("ACQ:STOP")
    result = []
    for i in chan:
        osc.write(f"ACQ:SOUR{i}:DATA?")
        # `#` starts block
        # `5` is length of data length
        # `32768` = 2*(2**14) is number of data bytes
        assert(osc.read_bytes(1 + 1 + 5) == b"#" b"5" b"32768")
        result.append(np.frombuffer(osc.read_bytes(32768), dtype=">h"))
        # alternative implementation with `struct` library
        #result.append(struct.unpack(f">{2**14}h", osc.read_bytes(32768)))
    
    return tuple(result)

# TODO: replace osc with specified acq
def voltage_scan(fg, osc, freq, n, start=0.5, stop=10, rev=False, sour=2, fmt="BIN", DEBUG=False):
    '''
    Use a function-generator to scan over a voltage range, at fixed frequency,
    for a single channel of an oscilloscope.
    Arguments:
        fg    -- function generator
        osc   -- oscilloscope
        freq  -- function generator frequency
        n     -- total number of voltage samples
        start -- start voltage
        stop  -- end voltage
        rev   -- boolean: perform scan in reverse?
        fmt   -- oscilloscope data format; ("ASCII" | "DEC")
        sour  -- source channel for function generator
    Returns:
        a dictionary of numpy arrays:
            V -- the input voltages
            s -- the oscilloscope trace at each voltage
    '''
    acq = acquire_binary if fmt == "BIN" else acquire_ascii
    
    scan_V = np.linspace(start, stop, n)
    scan_V = scan_V[::-1] if rev else scan_V
    scan_s = np.empty((n, 2**14), dtype="h" if fmt=="BIN" else "f")

    fg.write_ascii_values(f"SOUR{sour}:FREQ:FIX ", [freq])
    for i, v in tqdm(enumerate(scan_V)):
        fg.write_ascii_values(f"SOUR{sour}:VOLT ", [v])
        x = acq(osc)[0]
        scan_s[i,:] = x
        if DEBUG:
            plt.plot(x)
            plt.title(v)
            plt.show()
    
    fg.write(f"SOUR{sour}:VOLT MIN")
    
    return {"V" : scan_V, "s" : scan_s}

# TODO: replace osc with specified acq
def voltage_scan_multiple(fg, osc, freq, n, start=0.5, stop=10, iters=1, rev=False, sour=2, fmt="BIN", DEBUG=False):
    '''
    Use a function-generator to scan over a voltage range, at fixed frequency,
    with multiple acquisitions per voltage.
    Arguments:
        fg    -- function generator
        osc   -- oscilloscope
        freq  -- function generator frequency
        n     -- total number of voltage samples
        start -- start voltage
        stop  -- end voltage
        iters -- number of acquisitions per voltage sample
        rev   -- boolean: perform scan in reverse?
        sour  -- source channel for function generator
        fmt   -- oscilloscope data format; ("ASCII" | "DEC")
    Returns:
        a dictionary:
            V  -- numpy array of input voltages
            ss -- tuple of oscilloscope traces at each voltage
    '''
    acq = acquire_binary if fmt == "BIN" else acquire_ascii
    
    scan_V = np.linspace(start, stop, n)
    scan_V = scan_V[::-1] if rev else scan_V
    scan_ss = np.empty((n, iters, 2**14), dtype="h" if fmt=="BIN" else "f")
    
    fg.write_ascii_values(f"SOUR{sour}:FREQ:FIX ", [freq])
    for i, v in tqdm(enumerate(scan_V)):
        fg.write_ascii_values(f"SOUR{sour}:VOLT ", [v])
        for j in range(iters):
            s = acq(osc)[0]
            scan_ss[i,j,:] = s
            if DEBUG:
                plt.plot(s)
                plt.title(v)
                plt.show()
    
    fg.write(f"SOUR{sour}:VOLT MIN")
    
    return {"V" : scan_V, "ss" : scan_ss}

# TODO: replace osc with specified acq
def frequency_scan_2channels(fg, osc, volt, n, start=60_000, stop=150_000, sour=2, fmt="ASCII", DEBUG=False):
    '''
    Use a function-generator to scan over a frequency range, at fixed voltage,
    for two channels of an oscilloscope.
    Arguments:
        fg    -- function generator
        osc   -- oscilloscope
        volt  -- function generator voltage
        n     -- total number of frequency samples
        start -- start frequency
        stop  -- end frequency
        sour  -- source channel for function generator
        fmt   -- oscilloscope data format; ("ASCII" | "DEC")
    Returns:
        a dictionary:
            f  -- the input frequencies
            s1 -- the oscilloscope channel 1 trace
            s2 -- the oscilloscope channel 2 trace
    '''
    assert(fmt == "ASCII") # FIXME: this method has problems using binary data for some reason
    #acq = acquire_binary if fmt == "BIN" else acquire_ascii
    acq = acquire_ascii_2channels
    
    scan_f = np.linspace(start, stop, n)
    scan_s1 = np.empty((n, 2**14), dtype="h" if fmt=="BIN" else "f")
    scan_s2 = np.empty((n, 2**14), dtype="h" if fmt=="BIN" else "f")
    
    fg.write_ascii_values(f"SOUR{sour}:VOLT ", [volt])
    for i, f in enumerate(scan_f):
        fg.write_ascii_values(f"SOUR{sour}:FREQ:FIX ", [f])
        #scan_s1[i,:], scan_s2[i,:] = acq(osc)
        x, y = acq(osc)
        scan_s1[i,:], scan_s2[i,:] = x, y
        if DEBUG:
            plt.figure()
            plt.plot(x)
            plt.show()
            plt.figure()
            plt.plot(y)
            plt.show()
        
    fg.write(f"SOUR{sour}:VOLT MIN")
    
    return {"f" : scan_f, "s1" : scan_s1, "s2" : scan_s2}

def frequency_response_test_driver(rp):
    '''
    This is a test driver, which checks the frequency response of the STEMlab
    output channel and circuit by comparing a loopback with a signal passed 
    through the circuit.
    Arguments:
        rp -- the STEMlab device
    Effects:
        serialize test data, as .npz, to "freq_response" folder in separate 
        timestamped folder
    '''
    volt = 1.0
    fmt = "ASCII"
    
    config_rp_fg(rp)
    config_rp_osc(rp, fmt)

    scan = frequency_scan_2channels(rp, rp, volt, 50, start=10_000, stop=150_000, fmt=fmt)
    #dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    dt = time.strftime("%Y%m%d%H%M%S")
    os.mkdir(f"freq_response/{dt}")
    np.savez_compressed(f"freq_response/{dt}/freq_response.npz" , **scan)
    
    plt.plot(scan['f'],
                (np.max(scan['s1'], axis=1) - np.min(scan['s1'], axis=1)) 
              / (np.max(scan['s2'], axis=1) - np.min(scan['s2'], axis=1)))
    
    return

def multifrequency_voltage_scan_test_driver(rp):
    '''
    This is a test driver, which performs voltage scans at multiple different
    frequencies.
    Arguments:
        rp -- the STEMlab device
    Effects:
        serialize test data, as .npz, to "scans" folder in separate 
        timestamped folder
    '''
    fmt = "ASCII"
    config_rp_fg(rp)
    config_rp_osc(rp, fmt, dec=8)
    
    dt = time.strftime("%Y%m%d%H%M%S")
    os.mkdir(f"scans/{dt}")
    
    freq = np.linspace(30_000, 100_000, 10)
    n = 400
    for f in freq:
        start = time.time()
        scan = voltage_scan(rp, rp, f, n, start=0.01, stop=1.0, fmt=fmt, rev=False, DEBUG=False)
        np.savez_compressed(f"scans/{dt}/f{int(f)}.npz" , **scan)
        stop  = time.time()
        print(f"{int(stop - start)}s elapsed")

def multifrequency_voltage_scan_multiple_test_driver(rp):
    '''
    This is a test driver, which performs multiple voltage scans at multiple 
    different frequencies.
    Arguments:
        rp -- the STEMlab device
    Effects:
        serialize test data, as .npz, to "scans" folder in separate 
        timestamped folder
    '''
    fmt = "ASCII"
    config_rp_fg(rp)
    config_rp_osc(rp, fmt, dec=1)
    
    dt = time.strftime("%Y%m%d%H%M%S")
    os.mkdir(f"scans/{dt}")
    
    freq = [60_000] # use a single frequency
    n = 400
    for f in freq:
        start = time.time()
        scan = voltage_scan_multiple(rp, rp, f, n, start=0.09, stop=0.19, fmt=fmt, iters=8, rev=False)
        np.savez_compressed(f"scans/{dt}/f{int(f)}-1.npz" , **scan)
        scan = voltage_scan_multiple(rp, rp, f, n, start=0.25, stop=0.40, fmt=fmt, iters=8, rev=False)
        np.savez_compressed(f"scans/{dt}/f{int(f)}-2.npz" , **scan)
        scan = voltage_scan(rp, rp, f, n, start=0.05, stop=1.0, fmt=fmt, rev=True)
        np.savez_compressed(f"scans/{dt}/r{int(f)}.npz" , **scan)
        stop  = time.time()
        print(f"{int(stop - start)}s elapsed")

def main():
    rm = get_resource_manager()
    rp = get_rp(rm, "your device IP here")
    
    # call the drivers you want to use here
    multifrequency_voltage_scan_test_driver(rp)
    
    rp.close()
    rm.close()

if __name__ == "__main__":
    main()
