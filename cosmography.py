"""
Purpose: 
	The Cosmology class contains functions to calculate cosmic cosmography.


Dependencies:
	numpy, scipy, sys, types
	cosmology_parameters: sets cosmology parameter values (from github.com/elzurdo/cosmo)

Functions:
	omegam_z(redshift): returns the matter density OmegaM(z)
	Ez(redshift): returns the evolution parameter E(z)=H(z)/H_0
	hubble_z(redshift): similar to Ez. Should compare timing differences
	hubble_a(a): similar to hubble_z but uses a=1/(1+redshift) as input
	friedmann: Returns the Friedmann equation, evaluated at the given value of a, in units of Mpc/h
	.
	.
	.
	.... More documentation required ....
	

Comments:
	``retired" codes are old versions of function that have been updated (and should eventually be erased)
Created by:
	 Eyal Kazin, Swinburne University of Technology during Sept 2011 - February 2014
	 eyalkazin@gmail.com
"""


# integrating the Friedman Equation
import numpy as np
import scipy.integrate
from scipy.interpolate import interp1d
import sys
import types



class Cosmology:
	def __init__(self,params=None):
		import cosmology_parameters 
		cosmoparams=cosmology_parameters.Cosmo_Params(params=params)
		self.cosmoparams=cosmoparams.params

	def retired_friedmann(self,a):
		"""Returns the Friedmann equation, evaluated at the given value of a"""
		return 3000./(a*a*np.sqrt(self.omegaM/a**3 + self.omegaR/a**4 + self.omegaL))
		
	def omegam_z(self,redshift):
		"""
		Calculates the mass density OmegaM(z) = OmegaM(0)*(1_z)^3/E(z)^2
		"""
		Ez=np.sqrt(self.cosmoparams['OmegaM']*(1+redshift)**3+self.cosmoparams['OmegaL']+(1-self.cosmoparams['OmegaL']-self.cosmoparams['OmegaM'])*(1+redshift)**2)
		#print "Ez",Ez
		return self.cosmoparams['OmegaM']*(1./Ez**2)*(1+redshift)**3
		
	def Ez(self,redshift=None):# the evolutionary part of H(z), meaning H(z)/H(z=0)  #calculate_hubblez(self,redshift=None):
		if redshift == None: redshift=self.cosmoparams['redshift']
		# This does not include radiation
		#the_Ez1 = np.sqrt(self.cosmoparams['OmegaM']*(1.+redshift)**3+self.cosmoparams['OmegaL']*(1.+redshift)**( 3*(1+self.cosmoparams['weq'])) + (self.cosmoparams['OmegaK'])*(1.+redshift)**2)
		
		# This includes radiation
		the_Ez  = np.sqrt(self.cosmoparams['OmegaM']*(1.+redshift)**3+self.cosmoparams['OmegaR']*(1.+redshift)**4+self.cosmoparams['OmegaL']*(1.+redshift)**( 3*(1+self.cosmoparams['weq'])) + (self.cosmoparams['OmegaK'])*(1.+redshift)**2)
		#print "OmegaK in Ez: ", self.cosmoparams['OmegaK']
		#print the_Ez1
		#print "the_Ez", the_Ez
		#sys.exit(1)
		return the_Ez
		
	def hubble_z(self,z):
		"Same exact code as self.Ez consider retiring ...."
		a=1./(1+np.array(z))
		return self.hubble_a(a)
		
		
	def hubble_a(self,a):
		"""
		Returns H(z) in units of H0. 
		Meaning if H(z)=H0*E(z), where E(z) is the evolutionary factor,
		this routine returns E(z)
		"""
		#return np.sqrt(self.cosmoparams['OmegaM']*(1.+redshift)**3+self.cosmoparams['OmegaL']*(1.+redshift)**( 3*(1+self.cosmoparams['weq'])) + (self.cosmoparams['OmegaK'])*(1.+redshift)**2)
		#print self.cosmoparams['OmegaR'],self.cosmoparams['OmegaK']
		#print "bah"
		#sys.exit(1)
		#print self.cosmoparams['OmegaM'],self.cosmoparams['OmegaL'],self.cosmoparams['weq'],self.cosmoparams['OmegaK'],self.cosmoparams['OmegaR'],self.cosmoparams['wgamma']
		#print "hubble_a"
		#sys.exit(1)
		#print "OmegaK in hubble_a: ", self.cosmoparams['OmegaK']
		return np.sqrt(self.cosmoparams['OmegaM']/a**3+self.cosmoparams['OmegaL']/a**( 3*(1+self.cosmoparams['weq'])) + (self.cosmoparams['OmegaK'])/a**2+ self.cosmoparams['OmegaR']/a**4)
	
	def friedmann(self,a):
		"""Returns the Friedmann equation, evaluated at the given value of a, in units of Mpc/h"""
		return self.cosmoparams['speed of light']/self.hubble_a(a)/100./a**2 # The 100. factor is assuming H0=100*h

	def ComovingDistance(self,z,verbose=False):
		""" Return the comoving distance between now and the given redshift.
		Returns answer in h^-1 Mpc)
		z must be positive, and a numpy.array!  """
		#import scipy.integrate
		#from scipy.interpolate import interp1d
		a1=1.0
		a0=1.0/(1.0+z)
		#print type(z)
		#sys.exit(1)
		if len(z)<10: 
			return np.array([scipy.integrate.quad(self.friedmann,a,a1)[0] for a in a0])
		if len(z)>=10:
			if verbose: print 'mucho z!'
			a0min=min(a0)
			na=100
			da=(a1-a0min)*1./(na-1)
			a_array=np.arange(na)*da+a0min	
			comov_few=np.array([scipy.integrate.quad(self.friedmann,a,a1)[0] for a in a_array])
			comov_interpol= interp1d(a_array,comov_few)
			return comov_interpol(a0)
			
	def angular_diameter_distance(self,z,verbose=False):
		"""
		return the proper (physical) angular diameter distance 
		in units of Mpc
		including curvature
		"""
		
		#print types.NotImplementedType
		#print "boooya"
		#sys.exit(1)
		if type(z) is types.FloatType:
			chi_h=self.ComovingDistance(np.array([z])) # comoving in Mpc/h
		else:
			try:
				if z.dtype=="float64":  # for numpy .... see: http://docs.scipy.org/doc/numpy/user/basics.types.html
					chi_h=self.ComovingDistance(np.array([z])) # comoving in Mpc/h
			except:
				chi_h=self.ComovingDistance(z,verbose=False) # comoving in Mpc/h
		#
		chi=chi_h*1./(self.cosmoparams["Hubble"]/100.) 
		#da= chi_h*1./(1+z)/(self.cosmoparams["Hubble"]/100.) # angular diameter distance (proper, physical) in Mpc
		
		if self.cosmoparams["OmegaK"]>0.:
			#print "angular_diameter_distance OmegaK>0"
			dh=self.cosmoparams["speed of light"]*1./self.cosmoparams["Hubble"]
			dm= dh*1./np.sqrt(self.cosmoparams["OmegaK"])*np.sinh(np.sqrt(self.cosmoparams["OmegaK"])*chi*1./dh)
		elif self.cosmoparams["OmegaK"]<0.:
			#print "angular_diameter_distance OmegaK<0"
			dh=self.cosmoparams["speed of light"]*1./self.cosmoparams["Hubble"]
			dm= dh*1./np.sqrt(-self.cosmoparams["OmegaK"])*np.sin(np.sqrt(-self.cosmoparams["OmegaK"])*chi*1./dh)
		else:
			#print "flat!" 
			dm=chi
		
		da=dm*1./(1+z)
		#print self.cosmoparams['OmegaM'],self.cosmoparams['OmegaL'],self.cosmoparams['weq'],self.cosmoparams['OmegaK'],self.cosmoparams['OmegaR'],self.cosmoparams['wgamma'],self.cosmoparams['Hubble']
		#print "Da", da, chi*1./(1+z)
		return da
	
	def dV_angle_average_distance(self,z):
		"""
		useful for testing 
		print (c.dV_angle_average_distance(z)), 3.e5*z/70.
		print print self.cosmoparams["speed of light"]*z*1./Hz, np.sqrt(((1.+z)**2.)*(dA**2.))
		"""
		Hz=self.hubble_z(z)*(self.cosmoparams["Hubble"]) # km/s/Mpc 
		dA=self.angular_diameter_distance(z)
		#print self.cosmoparams["speed of light"]*z*1./Hz, dA
		dV=((dA**2.)*self.cosmoparams["speed of light"]*z*((1.+z)**2)*1./Hz)**(1./3)
		return dV
	
	def distance_modulas(self,z,verbose=False):
		dL=((1+z)**2)*self.angular_diameter_distance(z,verbose=False) # Luminosity distance
		#dL_h=dL*(self.cosmoparams["Hubble"]/100.) # H0-independent Luminosity distance, as in Goliath, Amanulla, et al. (2001)  just under equation (2)
		####dL_h=dL*(self.cosmoparams["Hubble"]/100.)/(70./100.) # choosing an h of 0.7?? H0-independent Luminosity distance, as in Goliath, Amanulla, et al. (2001)  just under equation (2)
		dL_h=dL ####*(self.cosmoparams["Hubble"]/100.)/(70./100.) # choosing an h of 0.7?? H0-independent Luminosity distance, as in Goliath, Amanulla, et al. (2001)  just under equation (2)
		dist_mod=5.*np.log10(dL_h) + 25. ####-19.3081547178
		#dist_mod=5.*np.log10(dL) + 25.
		#print dL[0],dL_h[0], self.cosmoparams["Hubble"]/100.
		#print dL[1],dL_h[1], self.cosmoparams["Hubble"]/100.
		#sys.exit(1)
		return dist_mod
		
	def distance_priors_cmb(self):
		#mutual parameters
		#shdec=self.sound_horizon(epoch="decoupling")  # updates: self.epoch_of_decoupling()
		shdec_komatsu=self.sound_horizon_komatsu(epoch="decoupling") # updates: self.epoch_of_decoupling()
		dAdec=self.angular_diameter_distance(self.zdec)
		
		
		# the distance priors
		#lAdec=self.acoustic_scale(soundhorizon_zdec=shdec, dAdec=dAdec)
		lAdec=self.acoustic_scale(soundhorizon_zdec=shdec_komatsu, dAdec=dAdec)
		Rdec=self.shift_parameter_R(dAdec=dAdec)
		
		#print shdec, shdec_komatsu
		#print lAdec, self.acoustic_scale(soundhorizon_zdec=shdec_komatsu, dAdec=dAdec)
		#sys.exit(1)
		
		return [lAdec,Rdec,self.zdec]
	
	def acoustic_scale(self, soundhorizon_zdec=None, dAdec=None):
		"""
		Equation (65) from Komatsu (2009)
		"""
		if not soundhorizon_zdec:
			soundhorizon_zdec=self.sound_horizon_komatsu(epoch="decoupling") # updates: self.epoch_of_decoupling()
			print "attempting to call sound hoirzon. If this function is called from distance_priors_cmb, this call should not be happening!\n purpose stop in cosmography.py"
			sys.exit(1)
		if not dAdec:
			dAdec=self.angular_diameter_distance(self.zdec) # might be a minor bug if self.zdec is not defined ...
		#print dAdec,self.zdec,soundhorizon_zdec
		lA= (1+self.zdec)*np.pi*dAdec*1./soundhorizon_zdec
		
		return lA[0]
		
	def shift_parameter_R(self, dAdec=None):
		"""
		As appearing in Equatino (69) in Komatsu (2009), given by Bond et al. (1997)
		"""
		#self.epoch_of_decoupling()
		if not dAdec:
			dAdec=self.angular_diameter_distance(self.zdec) # might be a minor bug if self.zdec is not defined ..
		R=100.*np.sqrt(self.cosmoparams["wm"])/self.cosmoparams["speed of light"]*(1+self.zdec)*dAdec
		#print  self.cosmoparams['OmegaM'],self.cosmoparams['OmegaL'],self.cosmoparams['weq'],self.cosmoparams['OmegaK'],self.cosmoparams['OmegaR'],self.cosmoparams['wgamma'],self.cosmoparams['Hubble'],self.cosmoparams["wm"],self.cosmoparams['OmegaM']*((self.cosmoparams['Hubble']*0.01)**2)
		return R[0]
	
	def sound_horizon_komatsu(self,na=100, epoch="drag"):
		
		self.epoch_of_equality() # I do not think I need this ...
		if epoch=="drag":
			self.epoch_of_end_of_drag()
			da=(self.adrag-0.)*1./(na-1)
		elif epoch=="decoupling":
			self.epoch_of_decoupling()
			da=(self.adec-0.)*1./(na-1)
		else:
			print "the input epoch is not defined here: ", epoch
			print "please update code.\npurpose stop in cosmography.py"
			sys.exit(1)
		
		aa=np.arange(na)*da
		aa[0]+=0.000000001 #0.000000001
		"""
		#R_a = np.zeros(na)
		#Ez_a = np.zeros(na)
		#for ia in range(na): 
			#atemp =aa[ia]
			#R_a[ia]=0.75*atemp*self.cosmoparams["wb"]*1./self.cosmoparams["wgamma"] #from Dodelson, pg. 82       self.speed_of_sound(ztemp)	
			#Ez_a[ia]=self.hubble_a(atemp)
		"""
		#print self.cosmoparams["wb"]*1./self.cosmoparams["wgamma"],self.cosmoparams["wb"],self.cosmoparams["wgamma"]
		R_a=0.75*aa*self.cosmoparams["wb"]*1./self.cosmoparams["wgamma"] #from Dodelson, pg. 82
		Ez_a=self.hubble_a(aa)
		H_a=Ez_a*self.cosmoparams["Hubble"]
		speedofsound_a=1./np.sqrt(3.*(1.+R_a)) # Equation 3.66 from Dodelson, pg. 82
		
		#print self.cosmoparams["OmegaM"],self.cosmoparams["Hubble"],self.cosmoparams["wm"],self.cosmoparams["OmegaM"]*(self.cosmoparams["Hubble"]*0.01)**2, H_a[-1]*1./self.cosmoparams["Hubble"],self.cosmoparams["OmegaR"],self.cosmoparams["OmegaK"]
		
		#if epoch=="decoupling":
		#	print self.adec, np.mean(speedofsound_a)
		"""
		print speedofsound_a
		print 1./aa-1.
		print H_a
		print epoch, self.adec
		print aa
		"""
		#print self.cosmoparams
		#sys.exit(1)		
		under_integral= speedofsound_a*1./((aa**2)*H_a)
		
		s = scipy.integrate.simps( under_integral, aa) 
		s*=self.cosmoparams["speed of light"]
		
		return s
		
	def sound_horizon(self,na=100, epoch="drag"):
		"""
		From Blake & Glazebrook, Eq. 1
		"""
		self.epoch_of_equality() # needs to update self.aeq
		if epoch=="drag":
			self.epoch_of_end_of_drag()
			da=(self.adrag-0.)*1./(na-1)
		elif epoch=="decoupling":
			self.epoch_of_decoupling()
			da=(self.adec-0.)*1./(na-1)
		else:
			print "the input epoch is not defined here: ", epoch
			print "please update code.\npurpose stop in cosmography.py"
			sys.exit(1)
		
		aa=np.arange(na)*da
		aa[0]+=0.000000001
		
		speedofsound_a = np.zeros(na)
		for ia in range(na): 
			ztemp = 1./aa[ia] - 1.
			speedofsound_a[ia]=self.speed_of_sound(ztemp)
		speedofsound_a*=self.cosmoparams["speed of light"]
		
		under_integral = speedofsound_a*1./np.sqrt(aa + self.aeq)
		s = scipy.integrate.simps( under_integral, aa) 
		
		soundhorizon=s*np.sqrt(1./self.cosmoparams["wm"])/100.  # This last 100. comes from normalization of 1./H0/OmegaM**0.5 meaning 1./wm/100.
		
		#if self.units=="Mpc":  # both yield the same result
		#	soundhorizon=s*1./self.cosmo_params["Hubble"]/np.sqrt(self.cosmo_params["OmegaM"])
		#if self.units=="h-1Mpc":
		#	soundhorizon=s*np.sqrt(1./self.cosmo_params["wm"])/100.  # This last 100. comes from normalization of 1./H0/OmegaM**0.5 meaning 1./wm/100.
		
		return soundhorizon
		
	def epoch_of_equality(self):
		self.zeq=(2.5e4)*self.cosmoparams["wm"]*(self.cosmoparams["Theta27"]**(-4))
		self.aeq=1./(1+self.zeq)
		
	def epoch_of_end_of_drag(self):
		b1=0.313*(self.cosmoparams["wm"]**(-0.419))*(1+0.607*(self.cosmoparams["wm"]**0.674) )
		b2=0.238*(self.cosmoparams["wm"]**0.223)
		self.zdrag=1291*( (self.cosmoparams["wm"]**0.251)*1./(1+0.659*(self.cosmoparams["wm"]**0.828)) )*(1+b1*(self.cosmoparams["wb"]**b2) )
		#print self.zdrag
		self.adrag=1./(1.+self.zdrag)
	
	def epoch_of_decoupling(self): 
		"""
		Equation (66) from Komatsu (2009), as proposed by Hu & Sugiyama (1996)
		"""
		
		"""
		g1=( 0.0783*(self.cosmoparams["wb"]**(-0.238))  )*1./( 1.+ 39.5*(self.cosmoparams["wb"]**(0.763)))
		g2=( 0.560 )*1./( 1.+21.1*(self.cosmoparams["wb"]**(1.81)))
		
		self.zdec=1048.*( 1+ 0.00124*(self.cosmoparams["wb"]**(-0.738)))*( 1.+ g1*(self.cosmoparams["wm"]**g2))
		#"""
		#"""
		wb=self.cosmoparams["wb"]
		wm=self.cosmoparams["wm"]
		
		g1= (0.0783*(wb**(-0.238)))*1./(1+39.5*(wb**(0.763)))
		g2= 0.560/(1+21.1*(wb**1.81))
		
		self.zdec=1048.*( 1.+0.00124*(wb**(-0.738)))*(1+g1*(wm**g2))
		#"""
		self.adec=1./(1.+self.zdec)
		#print self.cosmoparams["wb"]
		#print self.zdec
		#sys.exit(1)
		#print self.cosmoparams["OmegaM"],self.cosmoparams["Hubble"],self.cosmoparams["wm"],self.cosmoparams["OmegaM"]*(self.cosmoparams["Hubble"]*0.01)**2,self.cosmoparams["wb"]
		
	def speed_of_sound(self,z):
		# In units of the speed of light. Meaning, you need to multiply result by the speed of sound.
		R=self.baryon_to_photo_ratio_z(z)
		#print "R ", R,
		cs=1./(3*(1+R))**0.5
		#print "cs ", cs,
		return cs
		
	def baryon_to_photo_ratio_z(self,z):
		"""
		Equation (5) in Einstein+Hu
		"""
		return 31.5*self.cosmoparams["wb"]*(1000./z)*(self.cosmoparams["Theta27"]**(-4))

	def nofz(self,redshifts,zbinsize=0.015,srad=12.566371, sqdeg=None):
		"""
		Calculates n(z) of a given distribution
		"""
		if sqdeg!=None:
			srad=sqdeg*( (np.pi/180.)**2 )
			print "survey of ", sqdeg," square degreees converted to ",
		print "survey sterrations ",srad
		nbins=(max(redshifts)-min(redshifts))*1./zbinsize
		[histvalues,borders]=np.histogram(redshifts,bins=nbins)
		zvals=np.zeros(len(histvalues))
		zslice_comovingvolumes=np.zeros(len(histvalues))
		for i in np.arange(len(histvalues)):
			zvals[i]=np.mean([borders[i],borders[i+1]])
			[rmin,rmax]=self.ComovingDistance(np.array([borders[i],borders[i+1]]))
			zslice_comovingvolumes[i]= 1./3*srad*(rmax**3-rmin**3)
		# need to update histvalues by volume !!!
		zdensity= histvalues*1./zslice_comovingvolumes
		return [zvals,zdensity]
