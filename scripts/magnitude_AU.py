#!/usr/bin/python3

#script to calculate a MLv (Magnitude Local,Vertical) earthquake magnitude using Geoscience Australia's WEST/EAST/SOUTH metric
https://auspass.edu.au/help/obspy_request.html

import numpy as np
from scipy.signal import welch
import obspy,math,sys
from obspy.geodetics import gps2dist_azimuth
from obspy import read, Stream, UTCDateTime,read_events
from obspy.core.event import Origin
from obspy.taup import TauPyModel
from obspy.clients.fdsn import Client
from obspy.signal.invsim import estimate_wood_anderson_amplitude_using_response


######################### DEFINE AN EARTHQUAKE

#sept 21 2021 victoria Mw 5.9 
event_lat=-37.486
event_long=146.347
event_depth=12.0
t=obspy.UTCDateTime('2021-09-21T23:15:53.617000Z')
region ="EAST" #can also be WEST or SOUTH (strictly Eyre Basin)

searchdist = 5 #distance in degrees from the event to add stations


#code to determine the maximum PSD frequency / time between peaks (for use in estimate_wood_anderson)
def get_peakpeak_time_psd(tr):
    peak = np.argmax(tr.data)
    data = tr.data[peak-100:peak+100]
    f,psd = welch(data,nperseg=64,fs=tr.stats.sampling_rate,nfft=64)
    f = f[2:]; psd=psd[2:] #avoid 0
    maxfreq = f[np.argmax(psd)]
    timespan = 1/(2*maxfreq)
    return timespan

#https://github.com/GeoscienceAustralia/ga-mla
def GA_mag(ampl_mm,dist_km,region='WEST'):
    if region.upper() == "SOUTH":
        mag = math.log10(ampl_mm) + 1.1*math.log10(dist_km) + 0.0013*dist_km + 0.7
    elif region.upper() == "EAST":
        mag = math.log10(ampl_mm) + (1.34*math.log10(dist_km/100)) + (0.00055 * (dist_km - 100)) + 3.13
    else:
        mag = math.log10(ampl_mm) + 1.137*math.log10(dist_km) + 0.000657*dist_km + 0.66
    return mag



################################### set velocity model (not too important) and data centers
model = TauPyModel(model="iasp91")
iris = Client("IRIS")
auspass=Client('AUSPASS')
#################

#collect all station inventory within ~5 degrees of event
iris_inv = iris.get_stations(network='AU',latitude=event_lat,longitude=event_long,maxradius=searchdist,channel='BHZ',startbefore=t,endafter=t,level='response') #only Z channel for now
auspass_inv = auspass.get_stations(network='*',latitude=event_lat,longitude=event_long,maxradius=searchdist,channel='H*Z',startbefore=t,endafter=t,level='response')
inv = iris_inv.copy(); inv += auspass_inv

data = []

#go through the inventory, pull waveforms, calculate magnitude, store results
for nslc in inv.get_contents()['channels']:
    n,s,l,c = nslc.split('.')

    if n=="S1" and c[0] == "B": continue #skip the 10hz data
    sta_lat,sta_lon = inv.get_coordinates(nslc)['latitude'], inv.get_coordinates(nslc)['longitude']
    epi_dist, az, baz = gps2dist_azimuth(float(event_lat),float(event_long), sta_lat, sta_lon)
    epi_dist_km = epi_dist / 1000

    arrivals=model.get_travel_times(source_depth_in_km=float(event_depth),distance_in_degree=epi_dist_km/(111.19*math.cos(sta_lat*np.pi/180)),phase_list=['ttbasic'])
    p_arrival = t+ arrivals[0].time

    if nslc in iris_inv.get_contents()['channels']:
        try: st = iris.get_waveforms(n,s,l,c,starttime=t-10,endtime=t+120)
        except: 
            print("  ! no data for %s" % nslc)
            continue
    elif nslc in auspass_inv.get_contents()['channels']:
        try: st = auspass.get_waveforms(n,s,l,c,starttime=t-10,endtime=t+120)
        except: 
            print("  ! no data for %s" % nslc)
            continue

    st.detrend("demean"); st.detrend("linear")
    st.filter('bandpass', freqmin=1, freqmax=10)
    st.taper(0.05,type='hann', max_length=2, side='left')
    st.merge(fill_value=0)
    tr_z= st.select(component="Z")[0]

    peakpeakamp = 2*max(abs(tr_z.data))
    peakpeaktime = get_peakpeak_time_psd(tr_z)
    response = inv.get_response(nslc,t)

    ampl_z = estimate_wood_anderson_amplitude_using_response(response,peakpeakamp,peakpeaktime) #returns amplitude in mm

    r = math.sqrt(float(event_depth)**2+epi_dist_km**2)

    mag = GA_mag(ampl_z,r,region)
    data.append([nslc,ampl_z,epi_dist_km,mag])
    print("%s success! mag %.2f (distance = %.2fkm)" % (nslc,mag,epi_dist_km))

amplitudes = [ele[1] for ele in data]
mags = [ele[3] for ele in data]
print("\n>>> average magnitude= %.2f (std %.2f)" % (np.mean(mags),np.std(mags)))