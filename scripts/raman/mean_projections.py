#! /usr/bin/env python

import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import scipy.optimize as opt
import argparse
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from ml_raman.raman.gamma_modes_pristine import gamma_modes_pristine
from ml_raman.raman.dosdata import GeneralDOSData
import csv

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "supercell",
        help="size of the supercell of the form int int. Third argument is taken to be 1 since we are working with a 2D system.",
        nargs=2,
        type=int
    )
    parser.add_argument(
        "natoms_pc",
        help="number of atomes in primitive cell.",
        type=int
    )
    parser.add_argument(
        "iso_concentration",
        help="concentration of added isotope.",
        type = str
    )
    parser.add_argument(
        "--index_range",
        help="minimum and maximum index of phonons files. The file name is supposed to have the format $ix$i_$j_$k_phonons.h5, where $ix$i is the supercell size, $j is the isotope concentration and $k is the file index. If index_range is not given, script computes the raman spectra for the file $ix$i_$j_phonons.h5 ",
    nargs = 2,
    type = int
    )
    parser.add_argument(
        "--smearing",
        help="type of smearing. Either Gauss or Lorentz. Default is Gauss.",
        type = str,
        default = "Gauss"
    )    
    parser.add_argument(
        "width",
        help="smearing width. For Gaussian smearing, this c orresponds to \sigma. For Lorentzian smearing, this corresponds to \gamma.",
        type=float
    )
    parser.add_argument(
        "--save_proj", help="create a savefile for raman mean projections, with smearing.", default=False
    )
    parser.add_argument(
        "npts",
        help="number of sampled points.",
        type = int
    )    
    parser.add_argument(
        "output_name", help="name of the written figure file.", type=str
    )
    parser.add_argument(
        "--save_fit", help="saves amplitude, the full width at half maximum and center of raman spectrum.", default=False
    )
    return(parser)
   
def main(args):
    Mx, My = args.supercell
    natoms_pc = args.natoms_pc
    iso_conc = args.iso_concentration 
    npts = args.npts
    width = args.width
    smearing = args.smearing
    savename = (
            args.output_name
            if args.output_name.endswith(".jpg")
            else args.output_name + ".jpg"
            )
    natoms_sc = natoms_pc * Mx * My
    nmodes   = 3*natoms_sc
    #nmodes = 8000
    eig      = np.empty((natoms_pc, nmodes))
    eig_avg  = np.zeros((natoms_pc, nmodes))
    raman_proj = np.empty((2,npts))
 
    #Define two gamma modes of pristine graphene
    v1, v2 = gamma_modes_pristine(natoms_sc)
    if args.index_range == None:
        
        f1 = h5py.File('/home/dounia/projects/rrg-cotemich-ac/dounia/datasets/raman/gra_isotopes/phonons'+str(Mx)+'x'+str(My)+'_'+iso_conc+'_phonons'+'.h5','r')
        key_energies = list(f1.keys())[0]
        key_modes = list(f1.keys())[1]
        # Get the data
        eigenval = list(f1[key_energies])
        eigvec = list(f1[key_modes])
        abs1 = np.inner(v1,np.transpose(eigvec))
        abs2 = np.inner(v2,np.transpose(eigvec))
        raman = abs1**2+abs2**2
        eig_avg[0] = eigenval
        eig_avg[1] = raman
        print("sum of raman projections on first mode = ", np.sum(abs1**2))
        print("sum of raman projections on second mode = ", np.sum(abs2**2))
        print("sum of raman projections = ", np.sum(raman))
        rdos = GeneralDOSData(eig_avg[0], eig_avg[1], info={"label":"raman"})  
    else:    
        idx_min, idx_max = args.index_range
        nindex = idx_max - idx_min +1
        eig_raman_sorted_tot = []
        for j in range(idx_min,idx_max+1):
            f1=h5py.File('/home/dounia/projects/rrg-cotemich-ac/dounia/datasets/raman/gra_isotopes/phonons/'+str(Mx)+'x'+str(My)+'_'+iso_conc+'_'+str(j)+'_phonons'+'.h5','r')
            key_energies = list(f1.keys())[0]
            key_modes = list(f1.keys())[1]
            # Get the data
            eigenval = np.array(f1[key_energies])
            eigvec = np.array(f1[key_modes])
            abs1 = np.inner(v1,np.transpose(eigvec))
            abs2 = np.inner(v2,np.transpose(eigvec))
            raman = abs1**2+abs2**2
            eig[0] = eigenval
            eig[1] = raman
            eig_raman_sorted = np.vstack((eigenval, raman))[:, eigenval.argsort()]
            #eigvec = np.zeros((nmodes,nmodes))
            eig_sort = eig[:, eig[0].argsort()] 
            eig_avg[0] = eig_avg[0] + eig_sort [0]
            eig_avg[1] = eig_avg[1] + eig_sort [1]
            #print(np.shape(eig_raman_sorted_tot))
            print("sum of raman projections on first mode = ", np.sum(abs1**2))
            print("sum of raman projections on second mode = ", np.sum(abs2**2))
            print("sum of raman projections = ", np.sum(raman))
            eig_raman_sorted_tot.append(eig_raman_sorted)
            f1.close()

        eig_avg = eig_avg/(nindex)
               
        rdos = GeneralDOSData(eig_avg[0], eig_avg[1], info={"label":"raman"})
        
    rfig = plt.figure() 
    #rdosax = rfig.add_axes([0.5, 0.2, 0.35, 0.7])
    rdosax = rfig.add_axes([0.2, 0.2, 0.75, 0.7])
    if width ==0:
        rdosax.plot(eig_avg[0], eig_avg[1], label="raman")
         # inset axes....
        x1, x2, y1, y2 = 1000, 1750, 0, 0.0001  # subregion of the original image
        axins = rdosax.inset_axes([0.2, 0.1, 0.35, 0.5], xlim=(x1, x2), ylim=(y1, y2))
        axins.plot(eig_avg[0], eig_avg[1], label="raman")
        rdosax.indicate_inset_zoom(axins, edgecolor="black")
    else:
        # rdosax = rdos.plot(npts=npts, width=width, ax=rdosax, xmin=1200, xmax=1800)
        rdos.plot(npts=npts, width=width, ax=rdosax, smearing = smearing)
        # inset axes....
        x1, x2, y1, y2 = 1000, 1750, 0, 0.0001  # subregion of the original image
        axins = rdosax.inset_axes([0.2, 0.1, 0.35, 0.5], xlim=(x1, x2), ylim=(y1, y2))
        rdos.plot(npts=npts, width=width, ax=axins)
        rdosax.indicate_inset_zoom(axins, edgecolor="black")

    rdosax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    #rdosax.set_xlim(1200,1800)
    rdosax.set_xlim(0,2000)
    rdosax.set_ylim(0,max(eig_avg[1]))
    #rdosax.set_ylim(0,0.0001)
    rdosax.set_title("Raman spectrum for "+str(Mx)+"x"+str(My)+" graphene sc at isotope conc "+str(iso_conc), fontsize=8)
    rdosax.set_xlabel("Frequency $\mathregular{(cm^{-1}}$)", fontsize=16)
    rdosax.set_ylabel("Intensity (a.u)", fontsize=16) 
    #rdosplot = rdos.sample_grid(npts=npts, width=width, xmin=1200, xmax=1800)
    #raman_proj[1] = rdosplot.get_weights()
    #raman_proj[0] = rdosplot.get_energies()
    #peaks, _ = find_peaks(raman, width=8)
    #peaks, _ = find_peaks(raman, height=0.01, width=1)
    #peaks,_ = find_peaks(raman_proj[1], height=0.05)
   # rdosax.plot(eigenval[int(peaks[0])],raman[int(peaks[0])],marker='.',label='peak at '+str('%.3f'%(eigenval[int(peaks[0])]))+' $\mathregular{cm^{-1}}$', markersize=10)
    #rdosax.plot(raman_proj[0][int(peaks[0])],raman_proj[1][int(peaks[0])],marker='.',label='peak at '+str('%.3f'%(raman_proj[0][int(peaks[0])]))+' $\mathregular{cm^{-1}}$', markersize=10)

    savefile=args.save_proj
    if savefile:
        if width ==0:
            rdosax.plot(eig_avg[0], eig_avg[1], label="raman")
            np.savetxt('raman'+args.output_name+'.dat', np.transpose(eig_avg))
        else:
            rdosplot = rdos.sample_grid(npts=npts, width=width, xmin=1200, xmax=1800, smearing = smearing)
            raman_proj[1] = rdosplot.get_weights()
            raman_proj[0] = rdosplot.get_energies()
            np.savetxt('raman'+args.output_name+'.dat', np.transpose(raman_proj))

    save_fit = args.save_fit
    if save_fit and width !=0:
        if smearing == 'Lorentz':
        # def _2Lorentzian(x, amp1, cen1, wid1, amp2,cen2,wid2):
                #return (amp1*wid1**2/((x-cen1)**2+wid1**2)) +\
                #           (amp2*wid2**2/((x-cen2)**2+wid2**2)) 
            def _1Lorentzian(x, amp1, cen1, wid1):
                return (amp1*wid1**2/((x-cen1)**2+wid1**2)) 
            popt_1lorentzian_total = [] 

            for i in range(len(eig_raman_sorted_tot)):
                rdos_i = GeneralDOSData(eig_raman_sorted_tot[i][0], eig_raman_sorted_tot[i][1], info={"label":"raman"})
                rdosplot_i = rdos_i.sample_grid(npts=npts, width=width, xmin=1200, xmax=1800, smearing = smearing)
                raman_weights_i = rdosplot_i.get_weights()
                raman_energies_i = rdosplot_i.get_energies()
                raman_select_weights_i = raman_weights_i[(1250<=raman_energies_i) & (raman_energies_i<=1600)]
                raman_select_energies_i = raman_energies_i[(1250<=raman_energies_i) & (raman_energies_i<=1600)]
                popt_1lorentzian_i, _ = curve_fit(_1Lorentzian, raman_select_energies_i, raman_select_weights_i, p0=[1, 1555,12])               
                popt_1lorentzian_total.append(popt_1lorentzian_i)

            popt_1lorentzian_mean = np.mean(popt_1lorentzian_total, axis = 0)
            popt_1lorentzian_std = np.std(popt_1lorentzian_total, axis = 0)
            #with open(args.output_name+'_FWHM.txt', 'w') as f:
            #   f.write('{} {}'.format(iso_conc, FWHM))
            with open('lorentzian_fit_'+args.output_name+'.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                field = ["amplitude", "center","FWHM" ,'amplitude_err', 'center_err', "FWHM_err"]
                writer.writerow(field)
                amplitude = popt_1lorentzian_mean[0]
                center = popt_1lorentzian_mean[1]
                FWHM = 2*np.abs(popt_1lorentzian_mean[2])
                amplitude_err =  popt_1lorentzian_std[0]
                center_err = popt_1lorentzian_std[1]
                FWHM_err = 2*popt_1lorentzian_std[2]
                writer.writerow([amplitude, center, FWHM, amplitude_err, center_err, FWHM_err])
            
            print("Lorentzian amplitude Lanczos: {} ± {}".format(amplitude,  amplitude_err))
            print("Lorentzian center Lanczos: {} ± {}".format(center,  center_err))
            print("Lorentzian FWHM Lanczos: {} ± {}".format(FWHM,  FWHM_err))
            rdosplot = rdos.sample_grid(npts=npts, width=width, xmin=1200, xmax=1800, smearing = smearing)
            rdosax.set_ylim(0, max(rdosplot.get_weights()))
            print ('max = ',max(rdosplot.get_weights()))
            #rdosax.set_ylim(0,1)
            #rdosax.set_ylim(0,0.0001)
            rdosax.plot(np.linspace(1250,1800,50),_1Lorentzian(np.linspace(1250,1800,50),*popt_1lorentzian_mean), label="lorentzian")
            # inset axes....
            x1, x2, y1, y2 = 1000, 1750, 0, 0.0001  # subregion of the original image
            axins = rdosax.inset_axes([0.2, 0.1, 0.35, 0.5], xlim=(x1, x2), ylim=(y1, y2)) 
            rdos.plot(npts=npts, width=width, ax=axins) 
            axins.plot(np.linspace(1250,1800,50),_1Lorentzian(np.linspace(1250,1800,50),*popt_1lorentzian_mean))
            rdosax.indicate_inset_zoom(axins, edgecolor="black")
            
        elif smearing == 'Gauss':
            print ('True')
            def _1Gaussian(x, amp1,cen1,sigma1):
                return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))       
            popt_1gaussian_total = []
            for i in range(len(eig_raman_sorted_tot)):
                rdos_i = GeneralDOSData(eig_raman_sorted_tot[i][0], eig_raman_sorted_tot[i][1], info={"label":"raman"})
                rdosplot_i = rdos_i.sample_grid(npts=npts, width=width, xmin=1200, xmax=1800, smearing = smearing)
                raman_weights_i = rdosplot_i.get_weights()
                raman_energies_i = rdosplot_i.get_energies()
                raman_select_weights_i = raman_weights_i[(1250<=raman_energies_i) & (raman_energies_i<=1600)]
                raman_select_energies_i = raman_energies_i[(1250<=raman_energies_i) & (raman_energies_i<=1600)]

                popt_1gaussian_i,_ = curve_fit(_1Gaussian, raman_select_energies_i, raman_select_weights_i, p0=[1, 1555,12])
                popt_1gaussian_total.append(popt_1gaussian_i)
            #with open(args.output_name+'_FWHM.txt', 'w') as f:
            #   f.write('{} {}'.format(iso_conc, FWHM))
            popt_1gaussian_mean = np.mean(popt_1gaussian_total, axis = 0)
            popt_1gaussian_std = np.std(popt_1gaussian_total, axis = 0)
            with open('gaussian_fit_'+args.output_name+'.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                field = ["amplitude", "center","FWHM" ,'amplitude_err', 'center_err', "FWHM_err"]
                writer.writerow(field)
                amplitude = popt_1gaussian_mean[0]
                center =  popt_1gaussian_mean[1]
                FWHM = 2*np.sqrt(2*np.log(2))* np.abs(popt_1gaussian_mean[2])
                amplitude_err = popt_1gaussian_std[0]
                center_err =  popt_1gaussian_std[1]
                FWHM_err = 2*np.sqrt(2*np.log(2))* popt_1gaussian_std[2]
                writer.writerow([amplitude, center, FWHM, amplitude_err, center_err, FWHM_err])
           
            print("Gaussian amplitude: {} ± {}".format(amplitude,  amplitude_err))
            print("Gaussian center: {} ± {}".format(center,  center_err))
            print("Gaussian FWHM: {} ± {}".format(FWHM,  FWHM_err))
            #rdosax.set_ylim(0,max(raman_weights))
            #rdosax.set_ylim(0,1)
            rdosplot = rdos.sample_grid(npts=npts, width=width, xmin=1200, xmax=1800)
            rdosax.set_ylim(0, max(rdosplot.get_weights())) 
            rdosax.plot(np.linspace(1250,1800,50),_1Gaussian(np.linspace(1250,1800,50),*popt_1gaussian_mean), label="gaussian")
            # inset axes....
            x1, x2, y1, y2 = 1000, 1750, 0, 0.0001  # subregion of the original image 
            axins = rdosax.inset_axes([0.2, 0.1, 0.35, 0.5], xlim=(x1, x2), ylim=(y1, y2))
            rdos.plot(npts=npts, width=width, ax=axins)
            axins.plot(np.linspace(1250,1800,50),_1Gaussian(np.linspace(1250,1800,50),*popt_1gaussian_mean))
            rdosax.indicate_inset_zoom(axins, edgecolor="black")
       
    plt.legend()
    plt.savefig(savename)     


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
