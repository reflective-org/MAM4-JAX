#! /usr/bin/env python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from scipy import stats
from sys import exit

def axis_format(axes,fontheight,ptitle,xticks,x_lim,yticks,y_lim):
    axes.set_title(ptitle, fontsize=fontheight, pad=8)
    # turn off axes minor ticks
    axes.minorticks_off()
    # set x-axis tick label and its font
    axes.set_xticks( xticks )
    axes.set_xticklabels( xticks )
    for tick in axes.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontheight) # specify integer or one of preset strings, e.g.
                                            # tick.label.set_fontsize('x-small') 
        tick.label.set_rotation('horizontal')
    # set x-axis title
    axes.set_xlabel(r'$\rm log_{_{10}}(\Delta t)$', fontsize=fontheight, color='black', labelpad=2)
    # set x-axis limit
    axes.set_xlim(x_lim)

    # set y-axis log scale
#    axes[k].set_yscale('log')
#    axes[k].get_yaxis().set_major_formatter(mt.LogFormatterSciNotation())
#    locminy = mt.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8,1.),numticks=20)
#    axes[k].yaxis.set_minor_locator(locminy)
#    axes[k].yaxis.set_minor_formatter(mt.NullFormatter())

    # set y-axis tick label and its font
    axes.set_yticks( yticks )
    axes.set_yticklabels( yticks )
    for tick in axes.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontheight) # specify integer or one of preset strings, e.g.
                                            # tick.label.set_fontsize('x-small') 
        tick.label.set_rotation('horizontal')
    # set y-axis title
    axes.set_ylabel(r'$\rm log_{_{10}}(Relative\ Error)$', fontsize=fontheight, color='black', labelpad=4)
    # set y-axis limit
    axes.set_ylim(y_lim)

def plot_convergence(outpath, out_name, var_type, x_array, y_array, **kwargs):
    # Say, "the default sans-serif font is COMIC SANS"
    plt.rcParams['font.sans-serif'] = "Arial"
    # Then, "ALWAYS use sans-serif fonts"
    plt.rcParams['font.family'] = "sans-serif"
    # Change the plot property globally
    plt.rcParams['axes.linewidth']    = 1.
    plt.rcParams['xtick.major.width'] = 1.
    plt.rcParams['xtick.major.size']  = 6.
    plt.rcParams['xtick.major.pad']   = 4
    plt.rcParams['ytick.major.width'] = 1.
    plt.rcParams['ytick.major.size']  = 6.
    plt.rcParams['ytick.minor.size']  = 1.
    plt.rcParams['ytick.major.pad']   = 4
    
    # set default values
    x_label          = np.arange(0,5,1)
    y_label          = np.arange(-16,2,2)
    x_lim            = [0,4]
    y_lim            = None
    miss_y_lim       = False
    loc              = 'best'
    left             = 0
    right            = 1
    top              = 1
    bottom           = 0
    hspace           = 0.1
    wspace           = 0.1
    nrows            = 2
    ncols            = 2
    fontheight       = 6
    linethick        = 4
    mksize           = 8

    # check if user specifies input values
    if  kwargs is not None:
        for key, value in kwargs.items():
            if  key == "xlabel":
                x_label    = value
            if  key == "ylabel":
                y_label    = value
            if  key == "xlim":
                x_lim      = value
            if  key == "ylim":
                y_lim      = value
            if  key == "loc":
                loc        = value
            if  key == "left":
                left       = value
            if  key == "right":
                right      = value
            if  key == "top":
                top        = value
            if  key == "bottom":
                bottom     = value
            if  key == "wspace":
                wspace     = value
            if  key == "hspace":
                hspace     = value
            if  key == "ncols":
                ncols      = value
            if  key == "nrows":
                nrows      = value
            if  key == "fontheight":
                fontheight = value
            if  key == "porder":
                porder     = value
            if  key == "var_type":
                var_type   = value

    xx               = np.log10(x_array)
    if  var_type == "aer":
        yy           = np.log10(y_array)
        colors       = ['royalblue','lightseagreen','orchid'] # ['royalblue','lightseagreen','orchid','orangered']
        legend_label = ["Accumulation","Aitken","Coarse"]     # ["Accumulation","Aitken","Coarse","Primary carbon"]
        nplots       = 1
    elif var_type == "gas":
        yy           = np.log10(y_array)
        colors       = ['orangered']                          # ['orange']
        legend_label = [""]
        nplots       = 1
    else:
        print("Unrecognized variable type: %s" % var_type)
        exit

    # adjust the space/size of subplots
    plt.subplots_adjust(left=left,right=right,bottom=bottom,top=top,hspace=hspace,wspace=wspace)
    # make plots
    axes = []
    for k in np.arange(nplots):
        idx = k + 1
        axes.append(plt.subplot(nrows,ncols,idx,adjustable='box',autoscale_on=False))
        if  var_type == "aer":
            for j in np.arange(len(colors)):
                axes[k].plot( xx, yy[:,j], marker='o', markerfacecolor=colors[j], color=colors[j],
                              markersize=mksize, linewidth=linethick, label=legend_label[j] )
            ptitle = "Self-convergence test of "+out_name
        else:
            for j in np.arange(len(colors)):
                axes[k].plot( xx, yy, marker='o', markerfacecolor=colors[j], color=colors[j],
                              markersize=mksize, linewidth=linethick, label=legend_label[j] )
            ptitle = "Self-convergence test of "+out_name
        # set y-axis limit
        if  y_lim is None:
            y_lim = [np.floor(np.amin(yy)),np.ceil(np.amax(yy))]
            miss_y_lim = True
        # set axes format
        axis_format(axes[k],fontheight,ptitle,x_label,x_lim,y_label,y_lim)
        # add porder line(s)
        x_order     = np.arange(x_lim[0]+0.5,x_lim[0]+2.5,1)
        for j in np.arange(len(porder)):
            y_order = np.arange(y_lim[1]-0.5-porder[j],y_lim[1]-0.5+porder[j],porder[j]) 
            if  porder[j] == 1:
                axes[k].plot( x_order, y_order, 'k--', linewidth=linethick, label=r'$\rm 1^{st}-order$' )
            elif porder[j] == 2:
                axes[k].plot( x_order, y_order, 'k--', linewidth=linethick, label=r'$\rm 2^{nd}-order$' )
            elif porder[j] == 3:
                axes[k].plot( x_order, y_order, 'k--', linewidth=linethick, label=r'$\rm 3^{rd}-order$' )
            elif porder[j] == 4:
                axes[k].plot( x_order, y_order, 'k--', linewidth=linethick, label=r'$\rm 4^{th}-order$' )
            else:
                print("Unsupported porder: "+str(porder[j]))
                exit
        # set legend
        axes[k].legend(loc=loc, fontsize=fontheight*0.8, frameon=False)
        if  miss_y_lim:
            y_lim = None

    fig_name = outpath+out_name+'.png'
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.show()

def show_figures(outpath, left, right, top, bottom, hspace, wspace):
    from matplotlib.image import imread
    plt.subplots_adjust(left=left,right=right,top=top,bottom=bottom,hspace=hspace,wspace=wspace)
    plt.subplot(1,2,1)
    fname = outpath+"aer_RK4_convergence.png"
    image = imread(fname,format="png")
    plt.imshow(image,cmap='Greys_r')
    plt.axis('off')
    plt.subplot(1,2,2)
    fname = outpath+"gas_RK4_convergence.png"
    image = imread(fname,format="png")
    plt.imshow(image,cmap='Greys_r')
    plt.axis('off')
    plt.show()
    
def calc_slope(nc_file, tau, ftau):
    ds                = xr.open_dataset(nc_file)
    mam_soa           = ds['mam_soa']
    mam_soag          = ds['mam_soag']
    mode              = ds['mode'].size
    tsteps            = np.array(ds['tsteps'])
    nsteps            = tsteps.size
    
    xx                = np.log10(tsteps[:-1])
    yy                = np.empty([nsteps-1])
    slope_aer         = np.empty([mode])
    
    # find the closest dt to tau
    idx               = (np.abs(tsteps - tau*ftau)).argmin()
    if  tsteps[idx] > tau*ftau:
        idx           = idx + 1
    
    for i in np.arange(nsteps-1):
        yy[i]         = np.abs(mam_soag[i] - mam_soag[-1]) / mam_soag[-1]
    yy                = np.log10(yy)
    # only perform linear regression to dt < tau
    x                 = xx[idx:]
    y                 = yy[idx:]
    slope_gas, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print("slope of fitted line for SOAG:          %.5f" % slope_gas)
 
    for n in np.arange(mode):
        for i in np.arange(nsteps-1):
            yy[i]     = np.abs(mam_soa[n,i] - mam_soa[n,-1]) / mam_soa[n,-1]
        yy            = np.log10(yy)
        y             = yy[idx:]
        slope_aer[n], intercept, r_value, p_value, std_err = stats.linregress(x, y)
        print("slope of fitted line for SOA in mode %d: %.5f" % (n+1, slope_aer[n]))

    ds.close()
    return slope_gas, slope_aer

def calc_tau(nc_file):
    ds             = xr.open_dataset(nc_file)
    uptkaer_h2so4  = np.array(ds['uptkaer_h2so4'])
    tau            = 1. / ( uptkaer_h2so4 * 0.81 )
    print("\u03C4 of SOAG/SOA condensation: %.4e" % tau)
    ds.close()
    return tau

def issue_flag(nc_file, slope_gas, slope_aer, delta):
    ds   = xr.open_dataset(nc_file)
    mode = ds['mode'].size

    slope_gas = np.abs(slope_gas - 1.) <= delta
    slope_aer = np.abs(slope_aer - 1.) <= delta
    num = np.count_nonzero(slope_gas) + np.count_nonzero(slope_aer)
    if  num == mode + 1:
        print("PASS")
    else:
        print("FAIL")

    ds.close()
