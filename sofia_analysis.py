from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import os

import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

import matplotlib.gridspec as gridspec

cubes = ['004.00+02.35_W','004.00+10.35_W','004.00+18.35_W','004.00+26.35_W','004.00+34.35_W',
	 '012.00+02.35_W','012.00+10.35_W','012.00+18.35_W','012.00+26.35_W','012.00+34.35_W',
         '020.00+02.35_W','020.00+10.35_W','020.00+18.35_W','020.00+26.35_W','020.00+34.35_W',
         '028.00+02.35_W','028.00+10.35_W','028.00+18.35_W','028.00+26.35_W','028.00+34.35_W',
         '036.00+02.35_W','036.00+10.35_W','036.00+18.35_W','036.00+26.35_W','036.00+34.35_W',
         '044.00+02.35_W','044.00+10.35_W','044.00+18.35_W','044.00+26.35_W','044.00+34.35_W',
         '052.00+02.35_W','052.00+10.35_W','052.00+18.35_W','052.00+26.35_W','052.00+34.35_W',
         '060.00+02.35_W','060.00+10.35_W','060.00+18.35_W','060.00+26.35_W','060.00+34.35_W',
         '068.00+02.35_W','068.00+10.35_W','068.00+18.35_W','068.00+26.35_W','068.00+34.35_W',
         '076.00+02.35_W','076.00+10.35_W','076.00+18.35_W','076.00+26.35_W','076.00+34.35_W',
         '084.00+02.35_W','084.00+10.35_W','084.00+18.35_W','084.00+26.35_W','084.00+34.35_W',
         '092.00+02.35_W','092.00+10.35_W','092.00+18.35_W','092.00+26.35_W','092.00+34.35_W',
         '100.00+02.35_W','100.00+10.35_W','100.00+18.35_W','100.00+26.35_W','100.00+34.35_W',
         '108.00+02.35_W','108.00+10.35_W','108.00+18.35_W','108.00+26.35_W','108.00+34.35_W',
         '116.00+02.35_W','116.00+10.35_W','116.00+18.35_W','116.00+26.35_W','116.00+34.35_W',
         '124.00+02.35_W','124.00+10.35_W','124.00+18.35_W','124.00+26.35_W','124.00+34.35_W',
         '132.00+02.35_W','132.00+10.35_W','132.00+18.35_W','132.00+26.35_W','132.00+34.35_W',
         '140.00+02.35_W','140.00+10.35_W','140.00+18.35_W','140.00+26.35_W','140.00+34.35_W',
         '148.00+02.35_W','148.00+10.35_W','148.00+18.35_W','148.00+26.35_W','148.00+34.35_W',
         '156.00+02.35_W','156.00+10.35_W','156.00+18.35_W','156.00+26.35_W','156.00+34.35_W',
         '164.00+02.35_W','164.00+10.35_W','164.00+18.35_W','164.00+26.35_W','164.00+34.35_W',
         '172.00+02.35_W','172.00+10.35_W','172.00+18.35_W','172.00+26.35_W','172.00+34.35_W',
         '180.00+02.35_W','180.00+10.35_W','180.00+18.35_W','180.00+26.35_W','180.00+34.35_W',
         '188.00+02.35_W','188.00+10.35_W','188.00+18.35_W','188.00+26.35_W','188.00+34.35_W',
         '196.00+02.35_W','196.00+10.35_W','196.00+18.35_W','196.00+26.35_W','196.00+34.35_W',
         '204.00+02.35_W','204.00+10.35_W','204.00+18.35_W','204.00+26.35_W','204.00+34.35_W',
         '212.00+02.35_W','212.00+10.35_W','212.00+18.35_W','212.00+26.35_W','212.00+34.35_W',
         '220.00+02.35_W','220.00+10.35_W','220.00+18.35_W','220.00+26.35_W','220.00+34.35_W',
         '228.00+02.35_W','228.00+10.35_W','228.00+18.35_W','228.00+26.35_W','228.00+34.35_W',
         '236.00+02.35_W','236.00+10.35_W','236.00+18.35_W','236.00+26.35_W','236.00+34.35_W',
         '244.00+02.35_W','244.00+10.35_W','244.00+18.35_W','244.00+26.35_W','244.00+34.35_W',
         '252.00+02.35_W','252.00+10.35_W','252.00+18.35_W','252.00+26.35_W','252.00+34.35_W',
         '260.00+02.35_W','260.00+10.35_W','260.00+18.35_W','260.00+26.35_W','260.00+34.35_W',
         '268.00+02.35_W','268.00+10.35_W','268.00+18.35_W','268.00+26.35_W','268.00+34.35_W',
         '276.00+02.35_W','276.00+10.35_W','276.00+18.35_W','276.00+26.35_W','276.00+34.35_W',
         '284.00+02.35_W','284.00+10.35_W','284.00+18.35_W','284.00+26.35_W','284.00+34.35_W',
         '292.00+02.35_W','292.00+10.35_W','292.00+18.35_W','292.00+26.35_W','292.00+34.35_W',
         '300.00+02.35_W','300.00+10.35_W','300.00+18.35_W','300.00+26.35_W','300.00+34.35_W',
         '308.00+02.35_W','308.00+10.35_W','308.00+18.35_W','308.00+26.35_W','308.00+34.35_W',
         '316.00+02.35_W','316.00+10.35_W','316.00+18.35_W','316.00+26.35_W','316.00+34.35_W',
         '324.00+02.35_W','324.00+10.35_W','324.00+18.35_W','324.00+26.35_W','324.00+34.35_W',
         '332.00+02.35_W','332.00+10.35_W','332.00+18.35_W','332.00+26.35_W','332.00+34.35_W',
         '340.00+02.35_W','340.00+10.35_W','340.00+18.35_W','340.00+26.35_W','340.00+34.35_W',
         '348.00+02.35_W','348.00+10.35_W','348.00+18.35_W','348.00+26.35_W','348.00+34.35_W',
         '356.00+02.35_W','356.00+10.35_W','356.00+18.35_W','356.00+26.35_W','356.00+34.35_W']

params = ['id','id_old','x_geo','y_geo','z_geo','x','y','z',
	  'x_min','x_max','y_min','y_max','z_min','z_max',
	  'n_pix','n_chan','n_los','ell3s_maj','ell3s_min','ell3s_pa',
	  'ell_maj','ell_min','ell_pa','f_int','f_peak','f_wm50','kin_pa',
	  'rms','snr_int','w20','w50','wm50','ra','dec','vopt']

sourceunits = ['','',' [pixels]',' [pixels]',' [pixels]',' [pixels]',' [pixels]',' [pixels]',
               ' [pixels]',' [pixels]',' [pixels]',' [pixels]',' [pixels]',' [pixels]',
               '',' [channels]',' [pixels]',' [pixels]',' [pixels]',' [deg]',
               ' [pixels]',' [pixels]',' [deg]',' [K]',' [K]',' [K]',' [deg]',
               ' [K]','',' [channels]',' [channels]',' [channels]',' [deg]',' [deg]',' [m/s]']

plotunits = ['','',' [pixels]',' [pixels]',' [pixels]',' [pixels]',' [pixels]',' [pixels]',
             ' [pixels]',' [pixels]',' [pixels]',' [pixels]',' [pixels]',' [pixels]',
             '',' [channels]',' [pixels]',' [pixels]',' [pixels]',' [deg]',
             ' [pixels]',' [pixels]',' [deg]',' [K]',' [K]',' [K]',' [deg]',
             ' [K]','',' [km/s]',' [km/s]',' [km/s]',' [deg]',' [deg]',' [km/s]']

def make_report_plot(ra,dec,i,plotpath,overwrite=False):

    mpl.rcParams.update({'font.size':13})

    cubename = 'GALFA_HI_RA+DEC_'+str(ra).zfill(3)+'.00+'+str(dec).zfill(5)+'_W_UnsharpMask_r=30'
    path = 'DR2W/'+str(ra).zfill(3)+'.00+'+str(dec).zfill(5)+'_W/UnsharpMask_r=30/R/objects/'

    plotpath, dirs, files = os.walk(plotpath).next()
    if cubename+'_%i.pdf' % i in files and not overwrite:
        return
        
    objectfits = fits.open(path+cubename+'_%i.fits' % i)
    mom0 = fits.open(path+cubename+'_%i_mom0.fits' % i)
    index, velocity, spectrum = np.loadtxt(path+cubename+'_%i_spec.txt' % i,unpack=True)
    sourcetable = np.loadtxt(path[:-8]+cubename+'_cat.ascii',usecols=range(35))

    wcs = WCS(objectfits[0].header).celestial

    pl.clf()
    f = pl.figure(figsize=(5,6.1))
    gs = gridspec.GridSpec(2,1,height_ratios=[4,1])
    ax4 = pl.subplot(gs[0],projection=wcs)
    ax1 = pl.subplot(gs[1])

    ax1.plot(velocity/1000,spectrum)
    ax1.set_xlabel('Velocity [km/s]')
    if np.min(spectrum) >= 0:
        ax1.set_ylim(ymin=-0.1*np.max(spectrum))

    objectfitsdata = objectfits[0].data
    objectfitsdata[objectfitsdata < 0] = 0
    im4 = ax4.imshow(np.sum(objectfitsdata,axis=0),origin='lower',cmap=mpl.cm.gray,interpolation='nearest')
    ax4.coords[0].set_ticks(number=3,exclude_overlapping=True)
    ax4.coords[1].set_ticks(number=3,exclude_overlapping=True)

    f.colorbar(im4,ax=ax4,pad=0.05)
    ax4.set_title(str(ra*10**4 + dec*10**2 + i*10**(-3)))
    plotfile = '/home/cal/defelippis/Desktop/reportplots/'+cubename+'_%i.pdf' % i
    pl.savefig(plotfile)

    objectfits.close()
    mom0.close()
 
    return



def set_galactic_peak(ra, dec):
    cubename = 'GALFA_HI_RA+DEC_'+str(ra).zfill(3)+'.00+'+str(dec).zfill(5)+'_W.fits'
    cube = fits.open('DR2W/original/'+cubename)
    mean_T = np.zeros(cube[0].header['NAXIS3'])
    for i in range(mean_T.size):
        mean_T[i] = np.nanmean(cube[0].data[i])
    vels = ( np.arange(0,mean_T.size)-(cube[0].header['CRPIX3']-1) ) * cube[0].header['CDELT3'] / 1000 
    minvel = vels[(mean_T>1)*(vels>-200)][0]
    maxvel = vels[(mean_T>1)*(vels<200)][-1]
    return minvel, maxvel

def channels_to_velwidth(channel_array):
    return (channel_array-1) * 736.122839600 / 1000 # in km/s


def make_object_plot(ra, dec, i, plotpath, overwrite=False):
    cubename = 'GALFA_HI_RA+DEC_'+str(ra).zfill(3)+'.00+'+str(dec).zfill(5)+'_W_UnsharpMask_r=30'
    path = 'DR2W/'+str(ra).zfill(3)+'.00+'+str(dec).zfill(5)+'_W/UnsharpMask_r=30/R/objects/'

    plotpath, dirs, files = os.walk(plotpath).next()
    if cubename+'_%i.pdf' % i in files and not overwrite:
        return
        
    objectfits = fits.open(path+cubename+'_%i.fits' % i)
    mom0 = fits.open(path+cubename+'_%i_mom0.fits' % i)
    index, velocity, spectrum = np.loadtxt(path+cubename+'_%i_spec.txt' % i,unpack=True)
    sourcetable = np.loadtxt(path[:-8]+cubename+'_cat.ascii',usecols=range(35))

    wcs = WCS(objectfits[0].header).celestial

    pl.clf()
    f = pl.figure()
    ax1 = f.add_subplot(221)
    ax3 = f.add_subplot(223)
    ax3.axis('off')
    ax2 = f.add_subplot(222,projection=wcs)
    ax4 = f.add_subplot(224,projection=wcs)

    ax1.plot(velocity/1000,spectrum)
    ax1.set_xlabel('Velocity [km/s]')
    ax1.set_title(str(ra*10**4 + dec*10**2 + i*10**(-3)))
    if np.min(spectrum) >= 0:
        ax1.set_ylim(ymin=-0.1*np.max(spectrum))

    im2 = ax2.imshow(mom0[0].data,origin='lower',cmap=mpl.cm.gray,interpolation='nearest')
    ax2.coords[0].set_ticks(number=3,color='magenta',exclude_overlapping=True)
    ax2.coords[1].set_ticks(number=3,color='magenta',exclude_overlapping=True)
    ax2.coords[0].set_ticklabel_visible(False)
    ax2.coords[1].set_ticklabel_visible(False)
    overlay2 = ax2.get_coords_overlay('galactic')
    overlay2[0].set_ticks(number=5,color='cyan',exclude_overlapping=True)
    overlay2[1].set_ticks(number=5,color='cyan',exclude_overlapping=True)
    #overlay2[0].set_ticklabel_visible(True)
    #overlay2[1].set_ticklabel_visible(True)
    #overlay2[0].set_axislabel(r'$l$')
    #overlay2[1].set_axislabel(r'$b$')
    overlay2.grid(ls='dotted',linewidth=0.25,color='cyan')
    f.colorbar(im2,ax=ax2,pad=0.275)

    objectfitsdata = objectfits[0].data
    objectfitsdata[objectfitsdata < 0] = 0
    im4 = ax4.imshow(np.sum(objectfitsdata,axis=0),origin='lower',cmap=mpl.cm.gray,interpolation='nearest')
    ax4.coords[0].set_ticks(number=3,color='magenta',exclude_overlapping=True)
    ax4.coords[1].set_ticks(number=3,color='magenta',exclude_overlapping=True)
    ax4.coords[0].set_axislabel(r'RA ($l$)')
    ax4.coords[1].set_axislabel(r'Dec ($b$)')
    #overlay4 = ax4.get_coords_overlay('galactic')
    #overlay4[0].set_ticks(number=5,color='cyan',exclude_overlapping=True)
    #overlay4[1].set_ticks(number=5,color='cyan',exclude_overlapping=True)
    #overlay4.grid(ls='dotted',linewidth=0.25,color='cyan')
    f.colorbar(im4,ax=ax4,pad=0.275)

    ax3.text(0,0.5,'SNR = %f' % sourcetable[i-1,28])
    #pl.tight_layout(pad=0.5)
    plotfile = plotpath+cubename+'_%i.pdf' % i
    pl.savefig(plotfile)

    objectfits.close()
    mom0.close()
    return

def make_object_plots(sourcetablename,allsources=False,overwrite=False):
    plotpath = 'plots/HI/'
    sources = Table.read(sourcetablename)
    if not allsources:
        sources = sources[sources['yng'] == 'y']
        plotpath = 'plots/candidates/HI/'
    ras = (sources['id'] // 10**4).astype(int)
    decs = sources['id'] % 10**4 // 1 / 100
    ids = (sources['id'] * 10**3 % 10**3).astype(int)
    print('overwrite =',overwrite)
    for j in range(ids.size):
       # make_object_plot(ras[j],decs[j],ids[j],plotpath,overwrite=overwrite)
        make_report_plot(ras[j],decs[j],ids[j],plotpath,overwrite=overwrite)

    return


def find_good_objects(ra, dec, overwrite=False, makeobjectplots=False):
    cubename = 'GALFA_HI_RA+DEC_'+str(ra).zfill(3)+'.00+'+str(dec).zfill(5)+'_W.fits'
    objectspath = 'DR2W/'+str(ra).zfill(3)+'.00+'+str(dec).zfill(5)+'_W/UnsharpMask_r=30/R/objects'
    path, dirs, files = os.walk(objectspath).next()

    if 'goodobjects.txt' in files and not overwrite:
        print('goodobjects.txt exists for',(ra,dec))
        return

    path += '/'
    file_count = len(files) 
    file_count //= 7
    #path = path[2:]
    print(file_count)
    print(path)

    galacticpeak = set_galactic_peak(ra, dec)
    print(galacticpeak)

    sourcetable = np.loadtxt(path[:-8]+'GALFA_HI_RA+DEC_'+path[5:19]+'_'+path[20:36]+'_cat.ascii',usecols=range(35))
#    nchan = (sourcetable[:,15] < 70) # set max number of channels
    w50 = (channels_to_velwidth(sourcetable[:,30]) < 50) # set max w50 = 50 km/s
    ellmaj = (sourcetable[:,20] < 10) # set max size in spatial dimension (fitted ellipse)
    axisrat = (sourcetable[:,17]/sourcetable[:,18] < 1.5) # set maximum ellipticity
    snr = (sourcetable[:,28] > 20) # set minimum integrated SNR 

    #snr_sorted = np.argsort(sourcetable[:,28])
    #sorted_ids = sourcetable[:,0][snr_sorted[::-1]].astype(int)

    #counter = 0
    goodobjects = []
    
    for i in range(1,int(sourcetable[-1,0])):
    #    plots = pl.subplots(2, 2)
    #    f = plots[0]
    #    ax1, ax2 = plots[1][0]
    #    ax3, ax4 = plots[1][1]
    #    ax3.axis('off')
    #    ax1.set_title('Object %i, SNR = %f' % (i,sourcetable[i-1,28]))
    #    # moment 0 map
    #    mom0 = fits.open(path+'GALFA_HI_RA+DEC_'+path[0:14]+'_'+path[15:31]+'_%i_mom0.fits' % i)
    #    im2 = ax2.imshow(mom0[0].data,origin='lower',cmap=mpl.cm.gray,interpolation='nearest')
    #    ax2.set_xlabel('RA (pixels)')
    #    ax2.set_ylabel('Dec (pixels)')
    #    f.colorbar(im2,ax=ax2)
    #    mom0.close()

    #    index, velocity, spectrum = np.loadtxt(path+'GALFA_HI_RA+DEC_'+path[0:14]+'_'+path[15:31]+'_%i_spec.txt' % i,unpack=True)
    #    ax1.plot(velocity/1000,spectrum)
    #    ax1.set_xlabel('Velocity [km/s]')
    #    if np.min(spectrum) >= 0:
    #        ax1.set_ylim(ymin=-0.1*np.max(spectrum))

    #    zoom = fits.open(path+'GALFA_HI_RA+DEC_'+path[0:14]+'_'+path[15:31]+'_%i.fits' % i)
    #    #middle = zoom[0].data.shape[2] // 2
    #    im4 = ax4.imshow(np.sum(zoom[0].data,axis=0),origin='lower',cmap=mpl.cm.gray,interpolation='nearest')
    #    ax4.set_xlabel('RA (pixels)')
    #    ax4.set_ylabel('Dec (pixels)')
    #    f.colorbar(im4,ax=ax4)
    #    zoom.close()    
        index, velocity, spectrum = np.loadtxt(path+'GALFA_HI_RA+DEC_'+path[5:19]+'_'+path[20:36]+'_%i_spec.txt' % i,unpack=True)
        if makeobjectplots:
            zoom = fits.open(path+'GALFA_HI_RA+DEC_'+path[5:19]+'_'+path[20:36]+'_%i.fits' % i)
            wcs = WCS(zoom[0].header).celestial
            
            f = pl.figure()
            ax1 = f.add_subplot(221)
            ax3 = f.add_subplot(223)
            ax3.axis('off')
            ax2 = f.add_subplot(222,projection=wcs)
            ax4 = f.add_subplot(224,projection=wcs)
            ax1.set_title('Object %i, SNR = %f' % (i,sourcetable[i-1,28]))
            
            mom0 = fits.open(path+'GALFA_HI_RA+DEC_'+path[5:19]+'_'+path[20:36]+'_%i_mom0.fits' % i)
            im2 = ax2.imshow(mom0[0].data,origin='lower',cmap=mpl.cm.gray,interpolation='nearest')
            ax2.coords[0].set_ticks(number=3,color='magenta',exclude_overlapping=True)
            ax2.coords[1].set_ticks(number=3,color='magenta',exclude_overlapping=True)
            ax2.coords[0].set_ticklabel_visible(False)
            ax2.coords[1].set_ticklabel_visible(False)
            ax2.coords[0].set_axislabel('RA')
            ax2.coords[1].set_axislabel('Dec')
            overlay2 = ax2.get_coords_overlay('galactic')
            overlay2[0].set_ticks(number=5,color='cyan',exclude_overlapping=True)
            overlay2[1].set_ticks(number=5,color='cyan',exclude_overlapping=True)
            overlay2[0].set_ticklabel_visible(False)
            overlay2[1].set_ticklabel_visible(False)
            overlay2[0].set_axislabel(r'$l$')
            overlay2[1].set_axislabel(r'$b$')
            overlay2.grid(ls='dotted',linewidth=0.25,color='cyan')
            mom0.close()
            
            
            ax1.plot(velocity/1000,spectrum)
            ax1.set_xlabel('Velocity [km/s]')
            if np.min(spectrum) >= 0:
                ax1.set_ylim(ymin=-0.1*np.max(spectrum))

            im4 = ax4.imshow(np.sum(zoom[0].data,axis=0),origin='lower',cmap=mpl.cm.gray,interpolation='nearest')
            ax4.coords[0].set_ticks(number=3,color='magenta',exclude_overlapping=True)
            ax4.coords[1].set_ticks(number=3,color='magenta',exclude_overlapping=True)
            overlay4 = ax4.get_coords_overlay('galactic')
            overlay4[0].set_ticks(number=5,color='cyan',exclude_overlapping=True)
            overlay4[1].set_ticks(number=5,color='cyan',exclude_overlapping=True)
            overlay4.grid(ls='dotted',linewidth=0.25,color='cyan')
            zoom.close()
            
            f.colorbar(im2,ax=ax2,pad=0.15)
            f.colorbar(im4,ax=ax4,pad=0.25)


        exclude = ''
        suffix = ''
#        if not nchan[i-1]:
#            exclude += 'n_chan,\n'
        if not w50[i-1]:
            exclude += 'w50,\n'
        if not ellmaj[i-1]:
            exclude += 'ell_maj,\n'
        if not axisrat[i-1]:
            exclude += 'ell_maj/ell_min,\n'
        if not snr[i-1]:
            exclude += 'snr_int,\n'
        vmax = velocity[np.where(spectrum == np.max(spectrum))[0]]/1000
        if (vmax > 0 and vmax < galacticpeak[-1]) or (vmax < 0 and vmax > galacticpeak[0]):
            exclude += 'galactic peak'
        if exclude != '':
            #ax2.set_title(exclude,color='r')
            #ax3.text(0.1,0.5,exclude,color='red')
            #suffix += '_cut'
            #prefix = ''
            folder = 'cut/'
        else:
            folder = ''
            goodobjects.append(i-1)
            #prefix = str(counter)

        name = 'pdfs/'+folder+'GALFA_HI_RA+DEC_'+path[5:19]+'_'+path[20:36]+suffix+'_%i_plots.pdf' % i
        
        #pl.tight_layout(pad=0.25)
        #print(path+'pdfs/GALFA_HI_RA+DEC_'+path[0:14]+'_'+path[15:31]+'_%i_plots' % i +suffix+'.pdf')
        #pl.savefig(path+name)
        #counter += 1
        #pl.clf()
        
    ###### this is object number index, not the number itself (add 1 for that)
    np.savetxt(path+'goodobjects.txt',goodobjects)
    return
        

def make_correlation_plots(cubes=cubes,folder='R'):
    labels = []  # list of labels for axes 
    colors = mpl.cm.rainbow(np.linspace(0, 1, len(cubes)))

    sourcetables = len(cubes)*[[]]
    goodobjects = len(cubes)*[[]]
    sourcetables1 = len(cubes)*[[]]
    goodobjects1 = len(cubes)*[[]]
    for i in range(len(cubes)):
        filepath = 'DR2W/'+cubes[i]+'/UnsharpMask_r=30/'+folder+'/'
        sourcetables[i] = np.loadtxt(filepath+'GALFA_HI_RA+DEC_'+cubes[i]+'_UnsharpMask_r=30_cat.ascii',usecols=range(35))
        goodobjects[i] = np.loadtxt(filepath+'objects/goodobjects.txt').astype(int)
        #filepath1 = cubes[i]+'/UnsharpMask_r=30/Q/'
        #sourcetables1[i] = np.loadtxt(filepath1+'GALFA_HI_RA+DEC_'+cubes[i]+'_UnsharpMask_r=30_cat.ascii',usecols=range(35))
        #goodobjects1[i] = np.loadtxt(filepath1+'objects/goodobjects.txt').astype(int)        

    #pl.clf()
    #histdata = np.array([])
    #histdata1 = np.array([])
    #for i in range(len(cubes)):
    #    histdata = np.concatenate((histdata,sourcetables[i][goodobjects[i],30]))
    #    histdata1 = np.concatenate((histdata,sourcetables1[i][goodobjects1[i],30]))
    #pl.hist(channels_to_velwidth(histdata),bins=10,histtype='step',color='b',label='New kernels',linewidth=2)
    #pl.hist(channels_to_velwidth(histdata1),bins=10,histtype='step',color='r',label='Old kernels',linewidth=2)
    ##pl.ylim(0,30)
    #pl.xlabel('w50 [km/s]')
    #pl.legend(loc='upper right')
    #pl.savefig('plots/w50hist.pdf')
    

    if folder == 'Q':
        leotlocation = 303
    if folder == 'R':
        leotlocation = 297

    pl.clf()   
    for i in range(len(cubes)): 
        c = colors[i]
        if cubes[i] == '140.00+18.35_W':
            leot = (goodobjects[i] == leotlocation)
            pl.plot(sourcetables[i][goodobjects[i],34][~leot]/1000,sourcetables[i][goodobjects[i],28][~leot],'.',label=cubes[i],color=c)
            pl.plot(sourcetables[i][goodobjects[i],34][leot]/1000,sourcetables[i][goodobjects[i],28][leot],'*',markersize=14,color=c)
        else:
            pl.plot(sourcetables[i][goodobjects[i],34]/1000,sourcetables[i][goodobjects[i],28],'.',label=cubes[i],color=c)
    pl.gca().set_yscale('log')
    pl.legend(loc='lower right',fontsize=6)
    pl.xlabel('vopt [km/s]')
    pl.ylabel('snr_int')
    pl.axhline(y=20,color='black',linestyle='dashed')
    pl.savefig('plots/'+folder+'/snr_vs_vopt.pdf')


    pl.clf()   
    for i in range(len(cubes)): 
        c = colors[i]
        if cubes[i] == '140.00+18.35_W':
            leot = (goodobjects[i] == leotlocation)
            pl.plot(channels_to_velwidth(sourcetables[i][goodobjects[i],30][~leot]),sourcetables[i][goodobjects[i],28][~leot],'y.',
                    label=cubes[i],color=c)
            pl.plot(channels_to_velwidth(sourcetables[i][goodobjects[i],30][leot]),sourcetables[i][goodobjects[i],28][leot],'y*',
                    markersize=14,color=c)
        else:
            pl.plot(channels_to_velwidth(sourcetables[i][goodobjects[i],30]),sourcetables[i][goodobjects[i],28],'.',label=cubes[i],color=c)
    pl.gca().set_yscale('log')
    pl.legend(loc='lower right',fontsize=6)
    pl.xlabel('w50 [km/s]')
    pl.ylabel('snr_int')
    pl.xlim(0,40)
    pl.axhline(y=20,color='black',linestyle='dashed')
    pl.axvline(x=5,color='black',linestyle='dashed')
    pl.savefig('plots/'+folder+'/snr_vs_w50.pdf')

    pl.clf()   
    for i in range(len(cubes)): 
        c = colors[i]
        if cubes[i] == '140.00+18.35_W':
            leot = (goodobjects[i] == leotlocation)
            pl.plot(channels_to_velwidth(sourcetables[i][goodobjects[i],31][~leot]),sourcetables[i][goodobjects[i],28][~leot],'y.',
                    label=cubes[i],color=c)
            pl.plot(channels_to_velwidth(sourcetables[i][goodobjects[i],31][leot]),sourcetables[i][goodobjects[i],28][leot],'y*',
                    markersize=14,color=c)
        else:
            pl.plot(channels_to_velwidth(sourcetables[i][goodobjects[i],31]),sourcetables[i][goodobjects[i],28],'.',label=cubes[i],color=c)
    pl.gca().set_yscale('log')
    pl.legend(loc='lower right',fontsize=6)
    pl.xlabel('wm50 [km/s]')
    pl.ylabel('snr_int')
    pl.xlim(0,40)
    pl.axhline(y=20,color='black',linestyle='dashed')
    pl.axvline(x=5,color='black',linestyle='dashed')
    pl.savefig('plots/'+folder+'/snr_vs_wm50.pdf')


    pl.clf()   
    for i in range(len(cubes)): 
        c = colors[i]
        if cubes[i] == '140.00+18.35_W':
            leot = (goodobjects[i] == leotlocation)
            pl.plot(sourcetables[i][goodobjects[i],15][~leot],sourcetables[i][goodobjects[i],28][~leot],'y.',label=cubes[i],color=c)
            pl.plot(sourcetables[i][goodobjects[i],15][leot],sourcetables[i][goodobjects[i],28][leot],'y*',markersize=14,color=c)
        else:
            pl.plot(sourcetables[i][goodobjects[i],15],sourcetables[i][goodobjects[i],28],'.',label=cubes[i],color=c)
    pl.gca().set_yscale('log')
    pl.legend(loc='lower right',fontsize=6)
    pl.xlabel('nchan [channels]')
    pl.ylabel('snr_int')
    pl.axhline(y=20,color='black',linestyle='dashed')
    pl.axvline(x=8,color='black',linestyle='dashed')
    pl.savefig('plots/'+folder+'/snr_vs_nchan.pdf')


    pl.clf()   
    for i in range(len(cubes)): 
        c = colors[i]
        if cubes[i] == '140.00+18.35_W':
            leot = (goodobjects[i] == leotlocation)
            pl.plot(sourcetables[i][goodobjects[i],16][~leot],sourcetables[i][goodobjects[i],28][~leot],'y.',label=cubes[i],color=c)
            pl.plot(sourcetables[i][goodobjects[i],16][leot],sourcetables[i][goodobjects[i],28][leot],'y*',markersize=14,color=c)
        else:
            pl.plot(sourcetables[i][goodobjects[i],16],sourcetables[i][goodobjects[i],28],'.',label=cubes[i],color=c)
    pl.gca().set_yscale('log')
    pl.legend(loc='lower right',fontsize=6)
    pl.xlabel('nlos [pixels]')
    pl.ylabel('snr_int')
    pl.axhline(y=20,color='black',linestyle='dashed')
    pl.savefig('plots/'+folder+'/snr_vs_nlos.pdf')


    pl.clf()   
    for i in range(len(cubes)): 
        c = colors[i]
        if cubes[i] == '140.00+18.35_W':
            leot = (goodobjects[i] == leotlocation)
            pl.plot(sourcetables[i][goodobjects[i],16][~leot],sourcetables[i][goodobjects[i],15][~leot],'y.',label=cubes[i],color=c)
            pl.plot(sourcetables[i][goodobjects[i],16][leot],sourcetables[i][goodobjects[i],15][leot],'y*',markersize=14,color=c)
        else:
            pl.plot(sourcetables[i][goodobjects[i],16],sourcetables[i][goodobjects[i],15],'.',label=cubes[i],color=c)
    pl.gca().set_yscale('log')
    pl.gca().set_xscale('log')
    pl.legend(loc='lower right',fontsize=6)
    pl.xlabel('nlos [pixels]')
    pl.ylabel('nchan [channels]')
    pl.savefig('plots/'+folder+'/nchan_vs_nlos.pdf')


    pl.clf()   
    for i in range(len(cubes)): 
        c = colors[i]
        if cubes[i] == '140.00+18.35_W':
            leot = (goodobjects[i] == leotlocation)
            pl.plot(sourcetables[i][goodobjects[i],29][~leot]/sourcetables[i][goodobjects[i],30][~leot],
                    sourcetables[i][goodobjects[i],28][~leot],'y.',label=cubes[i],color=c)
            pl.plot(sourcetables[i][goodobjects[i],29][leot]/sourcetables[i][goodobjects[i],30][leot],
                    sourcetables[i][goodobjects[i],28][leot],'y*',markersize=14,color=c)
        else:
            pl.plot(sourcetables[i][goodobjects[i],29]/sourcetables[i][goodobjects[i],30],
                    sourcetables[i][goodobjects[i],28],'.',label=cubes[i],color=c)
    pl.gca().set_yscale('log')
    #pl.gca().set_xscale('log')
    
    pl.legend(loc='lower right',fontsize=6)
    pl.xlabel('w20/w50')
    pl.ylabel('snr_int')
    pl.axhline(y=20,color='black',linestyle='dashed')
    pl.savefig('plots/'+folder+'/snr_vs_w20overw50.pdf')


    pl.clf()   
    for i in range(len(cubes)): 
        c = colors[i]
        if cubes[i] == '140.00+18.35_W':
            leot = (goodobjects[i] == leotlocation)
            pl.plot(sourcetables[i][goodobjects[i],20][~leot],sourcetables[i][goodobjects[i],28][~leot],'y.',label=cubes[i],color=c)
            pl.plot(sourcetables[i][goodobjects[i],20][leot],sourcetables[i][goodobjects[i],28][leot],'y*',markersize=14,color=c)
        else:
            pl.plot(sourcetables[i][goodobjects[i],20],sourcetables[i][goodobjects[i],28],'.',label=cubes[i],color=c)
    pl.gca().set_yscale('log')
    pl.legend(loc='lower right',fontsize=6)
    pl.xlabel('ell_maj [pixels]')
    pl.ylabel('snr_int')
    pl.xlim(1,15)
    pl.axhline(y=20,color='black',linestyle='dashed')
    pl.savefig('plots/'+folder+'/snr_vs_ellmaj.pdf')

    pl.clf()   
    for i in range(len(cubes)): 
        c = colors[i]
        if cubes[i] == '140.00+18.35_W':
            leot = (goodobjects[i] == leotlocation)
            pl.plot(sourcetables[i][goodobjects[i],17][~leot],sourcetables[i][goodobjects[i],28][~leot],'y.',label=cubes[i],color=c)
            pl.plot(sourcetables[i][goodobjects[i],17][leot],sourcetables[i][goodobjects[i],28][leot],'y*',markersize=14,color=c)
        else:
            pl.plot(sourcetables[i][goodobjects[i],17],sourcetables[i][goodobjects[i],28],'.',label=cubes[i],color=c)
    pl.gca().set_yscale('log')
    pl.legend(loc='lower right',fontsize=6)
    pl.xlabel('ell3s_maj [pixels]')
    pl.ylabel('snr_int')
    pl.xlim(1,15)
    pl.axhline(y=20,color='black',linestyle='dashed')
    pl.savefig('plots/'+folder+'/snr_vs_ell3smaj.pdf')


    pl.clf()   
    for i in range(len(cubes)): 
        c = colors[i]
        ras, decs = sourcetables[i][goodobjects[i],32], sourcetables[i][goodobjects[i],33]
        coords = SkyCoord(ra=ras*u.degree, dec=decs*u.degree, frame='icrs')
        lats, longs = coords.galactic.b, coords.galactic.l
        if cubes[i] == '140.00+18.35_W':
            leot = (goodobjects[i] == leotlocation)
            #return lats, longs, leot
            pl.plot(longs[~leot],lats[~leot],'y.',label=cubes[i],color=c)
            pl.plot(longs[leot],lats[leot],'y*',markersize=14,color=c)
        else:
            pl.plot(longs,lats,'.',label=cubes[i],color=c)
    #pl.gca().set_yscale('log')
    pl.legend(loc='lower right',fontsize=6)
    pl.xlabel('galactic longitude [deg]')
    pl.ylabel('galactic lattitude [deg]')
    pl.savefig('plots/'+folder+'/lat_vs_long.pdf')


    pl.clf()   
    for i in range(len(cubes)): 
        c = colors[i]
        if cubes[i] == '140.00+18.35_W':
            leot = (goodobjects[i] == leotlocation)
            pl.plot(channels_to_velwidth(sourcetables[i][goodobjects[i],30][~leot]),sourcetables[i][goodobjects[i],34][~leot]/1000,'y.',
                    label=cubes[i],color=c)
            pl.plot(channels_to_velwidth(sourcetables[i][goodobjects[i],30][leot]),sourcetables[i][goodobjects[i],34][leot]/1000,'y*',
                    markersize=14,color=c)
        else:
            pl.plot(channels_to_velwidth(sourcetables[i][goodobjects[i],30]),sourcetables[i][goodobjects[i],34]/1000,'.',label=cubes[i],color=c)
    #pl.gca().set_yscale('log')
    pl.legend(loc='lower right',fontsize=6)
    pl.xlabel('w50 [km/s]')
    pl.ylabel('vopt [km/s]')
    pl.xlim(0,40)
    pl.axvline(x=5,color='black',linestyle='dashed')
    pl.savefig('plots/'+folder+'/vopt_vs_w50.pdf')

    pl.clf()   
    for i in range(len(cubes)): 
        c = colors[i]
        if cubes[i] == '140.00+18.35_W':
            leot = (goodobjects[i] == leotlocation)
            pl.plot(channels_to_velwidth(sourcetables[i][goodobjects[i],31][~leot]),sourcetables[i][goodobjects[i],34][~leot]/1000,'y.',
                    label=cubes[i],color=c)
            pl.plot(channels_to_velwidth(sourcetables[i][goodobjects[i],31][leot]),sourcetables[i][goodobjects[i],34][leot]/1000,'y*',
                    markersize=14,color=c)
        else:
            pl.plot(channels_to_velwidth(sourcetables[i][goodobjects[i],31]),sourcetables[i][goodobjects[i],34]/1000,'.',label=cubes[i],color=c)
    #pl.gca().set_yscale('log')
    pl.legend(loc='lower right',fontsize=6)
    pl.xlabel('wm50 [km/s]')
    pl.ylabel('vopt [km/s]')
    pl.xlim(0,40)
    pl.axvline(x=5,color='black',linestyle='dashed')
    pl.savefig('plots/'+folder+'/vopt_vs_wm50.pdf')


    pl.clf()   
    for i in range(len(cubes)): 
        c = colors[i]
        if cubes[i] == '140.00+18.35_W':
            leot = (goodobjects[i] == leotlocation)
            pl.plot(sourcetables[i][goodobjects[i],32][~leot],sourcetables[i][goodobjects[i],34][~leot]/1000,'y.',label=cubes[i],color=c)
            pl.plot(sourcetables[i][goodobjects[i],32][leot],sourcetables[i][goodobjects[i],34][leot]/1000,'y*',markersize=14,color=c)
        else:
            pl.plot(sourcetables[i][goodobjects[i],32],sourcetables[i][goodobjects[i],34]/1000,'.',label=cubes[i],color=c)
    #pl.gca().set_yscale('log')
    pl.legend(loc='lower right',fontsize=6)
    pl.xlabel('RA [deg]')
    pl.ylabel('vopt [km/s]')
    pl.savefig('plots/'+folder+'/vopt_vs_ra.pdf')
    return


def make_correlation_plot(x,y,z='',logx=False,logy=False,logz=False):
    xparam = params.index(x)
    yparam = params.index(y)
    xvalues = np.array([])
    yvalues = np.array([])
    dwarfvalues = np.array([])
    if z != '':
	zparam = params.index(z)
        zvalues = np.array([])
        cmap = mpl.cm.rainbow
    else:
        colors = mpl.cm.rainbow(np.linspace(0, 1, len(cubes)))

    if logz:
        norm = mpl.colors.LogNorm()
    else:
        norm = None
        
    sourcetables = len(cubes)*[[]]
    goodobjects = len(cubes)*[[]]
    for i in range(len(cubes)):
        filepath = 'DR2W/'+cubes[i]+'/UnsharpMask_r=30/R/'
        sourcetables[i] = np.loadtxt(filepath+'GALFA_HI_RA+DEC_'+cubes[i]+'_UnsharpMask_r=30_cat.ascii',usecols=range(35))
        sourcetables[i][:,29:32] = channels_to_velwidth(sourcetables[i][:,29:32])
        sourcetables[i][:,34] /= 1000
        goodobjects[i] = np.loadtxt(filepath+'objects/goodobjects.txt').astype(int)
    
    #if folder == 'Q':
    #    leotlocation = 303
    #if folder == 'R':
    leotlocation = 297

    pl.clf()
    for i in range(len(cubes)): 
        if cubes[i] == '140.00+18.35_W':
            dwarf = (goodobjects[i] == leotlocation)
        elif cubes[i] == '148.00+02.35_W':
            dwarf = (goodobjects[i] == 162)
        elif cubes[i] == '180.00+34.35_W':
            dwarf = (goodobjects[i] == 138)
        elif cubes[i] == '196.00+10.35_W':
            dwarf = (goodobjects[i] == 151)
        elif cubes[i] == '212.00+26.35_W':
            dwarf = (goodobjects[i] == 64)
        elif cubes[i] == '356.00+18.35_W':
            dwarf = (goodobjects[i] == 13)
        else:
            dwarf = (goodobjects[i] != goodobjects[i])

        xs = sourcetables[i][goodobjects[i],xparam]
        ys = sourcetables[i][goodobjects[i],yparam]
        ###### JUST FOR LAT VS LONG PLOT ######
        #coords = SkyCoord(ra=xs*u.degree, dec=ys*u.degree, frame='icrs')
        #xs, ys = coords.galactic.l, coords.galactic.b
        #######################################
        if z != '':
            dwarfvalues = np.concatenate((dwarfvalues, dwarf))
            xvalues = np.concatenate((xvalues,xs))
            yvalues = np.concatenate((yvalues,ys))
            zvalues = np.concatenate((zvalues,sourcetables[i][goodobjects[i],zparam]))
        else:
            color = colors[i]
            pl.scatter(xs[dwarf],ys[dwarf],marker='*',s=100,color=color,linewidth='1',edgecolor='black')
            pl.scatter(xs[~dwarf],ys[~dwarf],marker='.',label=cubes[i],color=color)
        
    dwarfvalues = dwarfvalues.astype(bool)
    plotname = params[yparam]+'_vs_'+params[xparam]
    #plotname = 'lat_vs_long'
    if z != '':
        pl.scatter(np.array(xvalues[~dwarfvalues]),np.array(yvalues[~dwarfvalues]),c=zvalues[~dwarfvalues],marker='.',s=10,
                   vmin=zvalues.min(),vmax=zvalues.max(),norm=norm,cmap=cmap)
        pl.scatter(np.array(xvalues[dwarfvalues]),np.array(yvalues[dwarfvalues]),c=zvalues[dwarfvalues],marker='*',s=50,linewidth='1',edgecolor='black',
                   vmin=zvalues.min(),vmax=zvalues.max(),norm=norm,cmap=cmap)
        cbar = pl.colorbar(format='%.1i')
    	cbar.set_label(params[zparam]+plotunits[zparam],rotation=90)
        plotname += '_z='+params[zparam]
    #else:
    #    pl.legend(fontsize=6)

    if logx:
        pl.gca().set_xscale('log')
    if logy:
        pl.gca().set_yscale('log')
    pl.xlabel(params[xparam]+plotunits[xparam])
    pl.ylabel(params[yparam]+plotunits[yparam])
    #pl.xlabel('galactic longitude'+plotunits[xparam])
    #pl.ylabel('galactic lattitude'+plotunits[yparam])

    pl.xlim(1,14)
    pl.ylim(0,50)
    pl.axvline(x=4,ymin=10/50.,color='black',linestyle='dashed',linewidth=1)#ymin=10,linewidth=2,color='k')
    pl.axvline(x=8,ymin=10/50.,color='black',linestyle='dashed',linewidth=1)
    pl.axhline(y=10,xmin=3/13,xmax=7/13,color='black',linestyle='dashed',linewidth=1)
    pl.axhline(y=50,xmin=3/13,xmax=7/13,color='black',linestyle='dashed',linewidth=1)

    pl.savefig('plots/correlations/'+plotname+'.pdf')
    return


def select_objects(criteria=[]):
    if len(criteria) == 0:
        return
    # criteria should be in format of [[parameter index, min, max],[parameter index, min, max],...]
    sources = np.zeros((1,35))
    names = np.array([])
    for i in range(len(cubes)):
        filepath = 'DR2W/'+cubes[i]+'/UnsharpMask_r=30/R/'
        sourcetables = np.loadtxt(filepath+'GALFA_HI_RA+DEC_'+cubes[i]+'_UnsharpMask_r=30_cat.ascii',usecols=range(35))
        sourcetables1 = np.copy(sourcetables)
        sourcetables1[:,29:32] = channels_to_velwidth(sourcetables[:,29:32])
        sourcetables1[:,34] /= 1000
        goodobjects = np.loadtxt(filepath+'objects/goodobjects.txt').astype(int)

        all_restrictions = (goodobjects == goodobjects)
        for var in criteria:
            restrict1 = sourcetables1[:,var[0]] > var[1]
            restrict2 = sourcetables1[:,var[0]] < var[2]
            all_restrictions *= restrict1[goodobjects]*restrict2[goodobjects]

        LeoT_like = goodobjects[all_restrictions]
        # names // 10**4 = RA
        # names % 10**4 // 1 / 100 = DEC
        # names * 10**3 % 10**3 = object id (first column of sourcetable)
        names = np.concatenate((names,
                                np.float(cubes[i][:6])*10**4 + np.float(cubes[i][7:12])*10**2 + sourcetables[LeoT_like,0]*10**(-3)))     
        sources = np.concatenate((sources,sourcetables[LeoT_like]))

    sources = np.delete(sources,0,0)
    sources[:,0] = names

    headerlist = []
    for i in range(len(params)):
        headerlist.append(params[i]+sourceunits[i])

    tab = Table(sources,names=headerlist)
    tab.write('LeoT_like_sources_'+str(len(cubes))+'cubes_new.csv',format='csv')

    #np.savetxt('LeoT_like_sources_'+str(len(cubes))+'cubes.csv',sources,delimiter=',',header=','.join(headerlist))
    return
