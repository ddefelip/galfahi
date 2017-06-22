from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import os

import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord


cubes = ['004.00+02.35_W','004.00+10.35_W',                                  '004.00+34.35_W',
	 '012.00+02.35_W',                                                   '012.00+34.35_W',
         '020.00+02.35_W','020.00+10.35_W',                 '020.00+26.35_W','020.00+34.35_W',
         '028.00+02.35_W',                                                   '028.00+34.35_W',
         '036.00+02.35_W',                                                   '036.00+34.35_W',
         '044.00+02.35_W','044.00+10.35_W','044.00+18.35_W','044.00+26.35_W','044.00+34.35_W',
         '052.00+02.35_W',                                                   '052.00+34.35_W',
         '060.00+02.35_W',                                                   '060.00+34.35_W',
         '068.00+02.35_W',                                                   '068.00+34.35_W',
         '076.00+02.35_W',                                                   '076.00+34.35_W',
         '084.00+02.35_W',                 '084.00+18.35_W',                 '084.00+34.35_W',
         '092.00+02.35_W',                                                   '092.00+34.35_W',
         '100.00+02.35_W',                                                   '100.00+34.35_W',

                                           '124.00+18.35_W','124.00+26.35_W','124.00+34.35_W',

         '140.00+02.35_W','140.00+10.35_W','140.00+18.35_W','140.00+26.35_W','140.00+34.35_W',

         '156.00+02.35_W','156.00+10.35_W','156.00+18.35_W','156.00+26.35_W','156.00+34.35_W',

                                           '236.00+18.35_W','236.00+26.35_W','236.00+34.35_W',
        
         '284.00+02.35_W',
       
         '332.00+02.35_W','332.00+10.35_W','332.00+18.35_W','332.00+26.35_W','332.00+34.35_W',
         '340.00+02.35_W',                                                   '340.00+34.35_W',
         '348.00+02.35_W',                                                   '348.00+34.35_W',
         '356.00+02.35_W',                                                   '356.00+34.35_W']

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


def set_galactic_peak(ra, dec):
    cubename = 'GALFA_HI_RA+DEC_'+str(ra).zfill(3)+'.00+'+str(dec).zfill(5)+'_W.fits'
    cube = fits.open('DR2W/'+cubename)
    mean_T = np.zeros(cube[0].header['NAXIS3'])
    for i in range(mean_T.size):
        mean_T[i] = np.nanmean(cube[0].data[i])
    vels = ( np.arange(0,mean_T.size)-(cube[0].header['CRPIX3']-1) ) * cube[0].header['CDELT3'] / 1000 
    minvel = vels[(mean_T>1)*(vels>-200)][0]
    maxvel = vels[(mean_T>1)*(vels<200)][-1]
    return minvel, maxvel

def channels_to_velwidth(channel_array):
    return (channel_array-1) * 736.122839600 / 1000 # in km/s


def make_object_plot(ra, dec, i):
    cubename = 'GALFA_HI_RA+DEC_'+str(ra).zfill(3)+'.00+'+str(dec).zfill(5)+'_W_UnsharpMask_r=30'
    path = 'DR2W/'+str(ra).zfill(3)+'.00+'+str(dec).zfill(5)+'_W/UnsharpMask_r=30/R/objects/'

    objectfits = fits.open(path+cubename+'_%i.fits' % i)
    mom0 = fits.open(path+cubename+'_%i_mom0.fits' % i)
    index, velocity, spectrum = np.loadtxt(path+cubename+'_%i_spec.txt' % i,unpack=True)
    sourcetable = np.loadtxt(path[:-8]+cubename+'_cat.ascii',usecols=range(35))

    #return objectfits
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
    plotfile = 'plots/objects/'+cubename+'_%i.pdf' % i
    pl.savefig(plotfile)

    objectfits.close()
    mom0.close()
    return

def make_object_plots(sourcetablename):
    delimiter = None
    if sourcetablename[-3:] == 'csv':
        delimiter = ','
    sources = np.loadtxt(sourcetablename,usecols=range(35),delimiter=delimiter)
    ras = (sources[:,0] // 10**4).astype(int)
    decs = sources[:,0] % 10**4 // 1 / 100
    ids = (sources[:,0] * 10**3 % 10**3).astype(int)
    for j in range(ids.size):
        make_object_plot(ras[j],decs[j],ids[j])
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

    #galacticpeaks = np.array([[-18,8], [-23,6], [-40, 11], [-55, 15], [-41, 31], [-43, 13], [-31,8], [-72, 116]])
    #for i in range(len(cubes)):
    #    if path[:14] == cubes[i]:
    #        galacticpeak = galacticpeaks[i]
    #        break

    galacticpeak = set_galactic_peak(ra, dec)
    print(galacticpeak)

    sourcetable = np.loadtxt(path[:-8]+'GALFA_HI_RA+DEC_'+path[5:19]+'_'+path[20:36]+'_cat.ascii',usecols=range(35))
    nchan = (sourcetable[:,15] < 70) # set max number of channels
    ellmaj = (sourcetable[:,20] < 10) # set max size in spatial dimension (fitted ellipse)
    axisrat = (sourcetable[:,20]/sourcetable[:,21] < 3) # set maximum ellipticity
    snr = (sourcetable[:,28] > 6) # set minimum integrated SNR 

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
        if not nchan[i-1]:
            exclude += 'n_chan,\n'
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
    leotvalues = np.array([])
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
            leot = (goodobjects[i] == leotlocation)
        else:
            leot = (goodobjects[i] != goodobjects[i])

        xs = sourcetables[i][goodobjects[i],xparam]
        ys = sourcetables[i][goodobjects[i],yparam]
        ###### JUST FOR LAT VS LONG PLOT ######
        #coords = SkyCoord(ra=xs*u.degree, dec=ys*u.degree, frame='icrs')
        #xs, ys = coords.galactic.l, coords.galactic.b
        #######################################
        if z != '':
            leotvalues = np.concatenate((leotvalues, leot))
            xvalues = np.concatenate((xvalues,xs))
            yvalues = np.concatenate((yvalues,ys))
            zvalues = np.concatenate((zvalues,sourcetables[i][goodobjects[i],zparam]))
        else:
            color = colors[i]
            pl.scatter(xs[leot],ys[leot],marker='*',s=100,color=color,linewidth='1',edgecolor='black')
            pl.scatter(xs[~leot],ys[~leot],marker='.',label=cubes[i],color=color)
        
    leotvalues = leotvalues.astype(bool)
            
    plotname = params[yparam]+'_vs_'+params[xparam]
    #plotname = 'lat_vs_long'
    if z != '':
        pl.scatter(np.array(xvalues[~leotvalues]),np.array(yvalues[~leotvalues]),c=zvalues[~leotvalues],marker='.',
                   vmin=zvalues.min(),vmax=zvalues.max(),norm=norm,cmap=cmap)
        pl.scatter(np.array(xvalues[leotvalues]),np.array(yvalues[leotvalues]),c=zvalues[leotvalues],marker='*',s=100,linewidth='1',edgecolor='black',
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

    ##pl.xlim(1,14)
    ##pl.ylim(0,35)
    ##pl.axvline(x=4.5,ymin=10/35.,color='black')#ymin=10,linewidth=2,color='k')
    ##pl.axvline(x=8,ymin=10/35.,color='black')
    ##pl.axhline(y=10,xmin=3.5/13,xmax=7./13,color='black')

    pl.savefig('plots/'+plotname+'.pdf')
    return


def select_objects(criteria=[]):
    if len(criteria) == 0:
        return
    # criteria should be in format of [[parameter index, min, max],[parameter index, min, max],...]
    #sourcetables = len(cubes)*[[]]
    #goodobjects = len(cubes)*[[]]
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
    
    #headerlist =  

    np.savetxt('LeoT_like_sources.csv',sources,delimiter=',',header=','.join(headerlist))
    return
