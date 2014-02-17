"""
Purpose: 
	Cosmo_Params  class: contains functions to evaluate and yield cosmological parameters.
	GrowthParams  class: contains functions to evaluate and yield  parameters of the growth of structure
	SoundHorizon  class: calculates the sound horizon in units of Mpc
	FittingParams class: yields parameters for modeling of the galaxy clustering correlation function.  

	.
	.
	.
	.... More documentation required ....
	

Comments:

Created by:
	 Eyal Kazin, Swinburne University of Technology during Sept 2011 - February 2014
	 eyalkazin@gmail.com
"""

class Cosmo_Params:
	"""
	Purpose: 
		Cosmo_Params class: contains functions to evaluate and yield cosmological parameters.


	Dependencies:
		sys

	Functions:
		empirical: generates self.params which contains parameter values used for analysis of various sky surveys (SDSS, WiggleZ and simulations)
		assign_paramers: assigns default values to self.params, if not previously given.
		.
		.
		.
		.... More documentation required ....
	

	Comments:
	"""

	def __init__(self,params=None,verbose=False):
		self.params=params
		self.verbose=verbose
		if type(params) == str: self.empirical()
		self.assign_parameters()
	
	def empirical(self):
		if self.verbose == True:
			print "\n\nempirical, cosmology_parameters.py: Going to use the cosmology: "+self.params+" \n\n"
		if self.params[0:8] == "LasDamas": 
			print "empirical, cosmology_parameters.py: cosmology_parameters.py: Using LasDamas input parameters"
			ld_params={'wm': 0.1225, 'wb': 0.0196,'OmegaK':0., 'Hubble':70.,'weq':-1.0,'ns':1.,'OmegaR':0.,'sigma8':0.8}
			if self.params == "LasDamas_0.16z0.36_sdss_nz": ld_params['redshift']=0.278
			if self.params == "LasDamas_0.16z0.44_sdss_nz": ld_params['redshift']=0.3145
			if self.params == "LasDamas_0.16z0.44_lightconed": ld_params['redshift']=0.33
			self.params=ld_params
		try:
			if self.params[0:5] == "SDSS_":
				print "empirical, cosmology_parameters.py: cosmology_parameters.py: Using SDSS input parameters"
				sdss_params={'wm': 0.1225, 'wb': 0.0196,'OmegaK':0., 'Hubble':70.,'weq':-1.0,'ns':0.963}
				if self.params == "SDSS_0.16z0.36": sdss_params['redshift']=0.278 
				if self.params == "SDSS_0.16z0.44": sdss_params['redshift']=0.314
				self.params=sdss_params
		except:
			None
		print self.params
		try:
			if self.params[0:7] == "WiggleZ":
				if self.params.__contains__("COLA"):
					print "empirical, cosmology_parameters.py: cosmology_parameters.py: Using WiZCOLA input parameters" # 0.2<z<0.6 b=1.0, 0.4<z<0.8 b=1.1, 0.6<z<1.0 b=1.2
					wigglez_params={'wm': 0.1357, 'wb': 0.0227,'OmegaK':0., 'Hubble':70.5,'weq':-1.0,'ns':0.960,'bias':1.1} # flat LCDM with OM=0.27 # GiggleZ should be 0.273
				else:
					print "empirical, cosmology_parameters.py: cosmology_parameters.py: Using WiggleZ input parameters" # 0.2<z<0.6 b=1.0, 0.4<z<0.8 b=1.1, 0.6<z<1.0 b=1.2
					wigglez_params={'wm': 0.1361, 'wb': 0.0226,'OmegaK':0., 'Hubble':71.,'weq':-1.0,'ns':0.963,'bias':1.1} # flat LCDM with OM=0.27 # GiggleZ should be 0.273
				
				
				"""
				need to calculate effective redshifts!
				"""
				wigglez_params["Amc"]=0.15 #0.43 #0.375
				wigglez_params["kstar"]=0.185 #0.18
				if self.params.__contains__("z0pt2_0pt6"):   wigglez_params["redshift"],wigglez_params["bias"],wigglez_params["norm0"], wigglez_params["kstar"]=0.44, 1.0, 1.57,  0.1169 #0.119,  #0.16
				elif self.params.__contains__("z0pt4_0pt8"): wigglez_params["redshift"],wigglez_params["bias"],wigglez_params["norm0"], wigglez_params["kstar"]=0.60, 1.1, 1.884, 0.1335 #0.126 #0.185
				elif self.params.__contains__("z0pt6_1pt0"): wigglez_params["redshift"],wigglez_params["bias"],wigglez_params["norm0"], wigglez_params["kstar"]=0.73, 1.2, 2.205, 0.14   #0.1425 #0.248
				
				if self.params.__contains__("Reconkstar"): 
					wigglez_params["norm0"]=wigglez_params["bias"]**2
					wigglez_params["kstar"]=0.39
				
				"""
				else:
					print "need to specify the WiggleZ params"
					print "purpose stop in cosmology_parameeters.py"
					print a
					sys.exit(1)
				"""
				self.params=wigglez_params
		except:
			None
		try:
			if self.params[0:7] == "GiggleZ":
			#if self.params.contains("GiggleZ"):
				print "empirical, cosmology_parameters.py: cosmology_parameters.py: Using GiggleZ input parameters"
				gigglez_params={'wm': 0.1376, 'wb': 0.0226,'OmegaK':0., 'Hubble':71.,'weq':-1.0,'ns':0.963}  # GiggleZ is 0.273
				"""
				need to calculate effective redshifts!
				"""
				gigglez_params["redshift"]=0.6
				if self.params=="GiggleZ cube": gigglez_params["redshift"]=0.0
				self.params=gigglez_params
		except:
			None
		try:
			if self.params[0:8] == "SPTHalos" or self.params[0:12] == "PTHalos BOSS":
				print "empirical, cosmology_parameters.py: Using SPTHalos input parameters"
				spthalo_params={'wm': 0.13426, 'wb': 0.0224,'OmegaK':0., 'Hubble':70.,'weq':-1.0,'ns':0.95,'sigma8':0.8,'OmegaR':0.,'redshift':0.57,'bias':2.0} # flat LCDM with OM=0.27 # GiggleZ should be 0.273
				"""
				need to calculate effective redshifts!
				"""
				self.params=spthalo_params
		except:
			None
			#LambdaCDM with Omega_m = 0.274, Omega_b * h2 = 0.0224 (this value is not mentioned in the paper). The other parameters are, I think n=0.95 and h=0.7,
		try: 
			if self.params == "Martin White Universe": # appears to be the same as SPTHalos cosmology
				print "cosmology_parameters.py: Using "+self.params+" input parameters"
				self.params={'wm': 0.13426, 'wb': 0.0224,'OmegaK':0., 'Hubble':70.,'weq':-1.0,'ns':0.95,'sigma8':0.8,'OmegaR':0.,'redshift':0.57,'bias':2.0} # flat LCDM with OM=0.27 # GiggleZ should be 0.273
		except:
			None
		"""
		if self.params[0:11] == "HorizonRun_":
		print "cosmology_parameters.py: Using LasDamas input parameters"
		hr_params={'wm': 0.134784, 'wb': 0.0228,'OmegaK':0., 'Hubble':72.,'weq':-1.0,'ns':0.96,'OmegaR':0,'sigma8'=0.8}
		near2, z=0.3418
		far2,  z=0.5300
		full,  z=0.44416
		"""
		#print 'redshift ',self.params['redshift']
		if self.verbose==True: print "meaning values: ", self.params
		
		if type(self.params) == str:
			print "====== bug ======"
			print "you entered parameter space: ", self.params
			print "which is not defined in empirical.cosmology_parameters.py"
			print "purpose break"
			a=b
			import sys
			sys.exit(1)
		
	
	def assign_parameters(self):
		#params_default={'redshift': 0.,'wm': 0.1323, 'wb': 0.02264, 'wneutrino':0.,'OmegaK':0.,'Hubble':70.,'weq':-1.0,'ns':0.963,'sigma8':0.8,'speed of light':299792.458,"Theta27": 2.725/2.7,"Neff":3.04,'wgamma':2.469e-5} # 'OmegaR':8.4e-5. 
		params_default={'redshift': 0.,'wm': 0.14195, 'wb': 0.02205, 'wneutrino':0.,'OmegaK':0.,'Hubble':67.3,'weq':-1.0,'ns':0.9603,'sigma8':0.8,'speed of light':299792.458,"Theta27": 2.725/2.7,"Neff":3.046,'wgamma':2.469e-5} # 'OmegaR':8.4e-5. Partially Table 2 from Planck1 paper 16 and part baseline cosmology 
		# I set 'wgamma':2.469e-5 due to T_CMB=2.725, as indicated in Komatsu et al. (2009), just above equation (7)
		if self.params != None:
			for key in params_default.keys():
				if not self.params.__contains__(key):
					if self.verbose: print "assign_parameters: cosmology_parameters.py: Updating "+key+" to ", params_default[key]
					self.params[key]=params_default[key]
				else:
					if self.verbose: print "assign_parameters: cosmology_parameters.py: Using input "+key+" value ", self.params[key]
		else: 
			if self.verbose: print "===== assign_parameters: cosmology_parameters.py: Using default cosmological parameters ======"
			self.params=params_default
		
		self.params["OmegaR"] = 	self.params['wgamma']*(1./(self.params['Hubble']/100.)**2)*(1.+0.2271*self.params['Neff'])
		self.params["wr"]=self.params["OmegaR"]*(self.params['Hubble']/100.)**2
		self.params['wcdm']=self.params['wm']-self.params['wb']
		self.params['OmegaM']=self.params['wm']*1./(self.params['Hubble']/100.)**2
		self.params['Omegab']=self.params['wb']*1./(self.params['Hubble']/100.)**2
		self.params['OmegaL']=1.-self.params['OmegaK']-self.params['OmegaM']-self.params['OmegaR']
		#print self.params["wr"],self.params["wm"],self.params["OmegaR"]
		#sys.exit(1)
		
class GrowthParams:
	"""
	Purpose: 
		GrowthParams class: contains functions to evaluate and yield  parameters of the growth of structure


	Dependencies:
		numpy, scipy, sys
		cosmography: used to evaluate (from github.com/elzurdo/cosmo)

	Functions:
		fgrowth_z(redshift): returns the logarithmic rate of growth of structure
		.
		.
		.
		.... More documentation required ....
	

	Comments:

	Created by:
		 Eyal Kazin, Swinburne University of Technology during Sept 2011 - February 2014
		 eyalkazin@gmail.com
	"""
	def __init__(self,cosmo_params=None):
		#self.cosmo_params=cosmo_params
		cosmo_params=Cosmo_Params(params=cosmo_params)
		self.cosmo_params=cosmo_params.params
	def fgrowth_z(self):
		import cosmography
		cosmo=cosmography.Cosmology(params=self.cosmo_params)
		self.fgrowth=cosmo.omegam_z(self.cosmo_params['redshift'])**0.55
		
	def growthDz(self, redshift=None,na=10000):
		import numpy as np
		import scipy.integrate
		
		if redshift == None: redshift=self.cosmo_params['redshift']
		#print "redshift used in growth: ", redshift
		hubble_a=self.Ez(redshift=redshift)  # in units of H0
		"""
		setting expansion factor array
		"""
		a=1./(1+redshift)
		da=(a-0)*1./(na-1)
		aa=np.arange(na)*da+0.
		
		denominator= (self.cosmo_params['OmegaM']*1.+self.cosmo_params['OmegaL']*(aa**(-3*self.cosmo_params['weq']))+(self.cosmo_params['OmegaK'])*aa)**1.5
		growthDz= scipy.integrate.simps( (aa**1.5)*1./denominator, aa) 
		growthDz*=5./2*self.cosmo_params['OmegaM']*hubble_a
		
		self.Dgrowth=growthDz
		return growthDz
		
	def Ez(self,redshift=None):# the evolutionary part of H(z), meaning H(z)/H(z=0)  #calculate_hubblez(self,redshift=None):
		import numpy as np
		if redshift == None: redshift=self.cosmo_params['redshift']
		return np.sqrt(self.cosmo_params['OmegaM']*(1.+redshift)**3+self.cosmo_params['OmegaL']*(1.+redshift)**( 3*(1+self.cosmo_params['weq'])) + (self.cosmo_params['OmegaK'])*(1.+redshift)**2)
		
		
class SoundHorizon:
	"""
	in units of Mpc
	"""
	
	def __init__(self,cosmo_params=None, units="Mpc"):
		cosmo_params=Cosmo_Params(params=cosmo_params)
		self.cosmo_params=cosmo_params.params
		self.cosmo_params["Theta27"] = 2.725/2.7
		self.units=units
		
		print "This SoundHorizon class has been retired"
		print "all functions within are now in cosmography.py"
		print "Update code!"
		print "purpose stop in cosmology_parameters.py"
		print a
		import sys
		sys.exit(1)
		
	def sound_horizon(self,na=1000):
		import numpy as np
		import scipy.integrate
		
		#arec=1./(1 + zrec)
		self.epoch_of_equality()
		self.epoch_of_end_of_drag()
		
		#print "a drag ", self.adrag, self.zdrag
		#print "a eq ", self.aeq, self.zeq
		
		da=(self.adrag-0)*1./(na-1)
		aa=np.arange(na)*da+0.0000001
		
		speedofsound_a = np.zeros(na)
		for ia in range(na): 
			ztemp = 1./aa[ia] - 1.
			speedofsound_a[ia]=self.speed_of_sound(ztemp)
			#print "ztemp", ztemp,
			#cs= self.speed_of_sound(ztemp)
			#print "cs2 ", cs
			#speedofsound_a[ia] = cs
		speedofsound_a*=self.cosmo_params["speed of light"]
		#print "\nspeed of light: ", self.cosmo_params["speed of light"]
		#print "speed of sound: ", speedofsound_a
		
		under_integral = speedofsound_a*1./np.sqrt(aa + self.aeq)
		s = scipy.integrate.simps( under_integral, aa) 
		
		soundhorizon=s*np.sqrt(1./self.cosmo_params["wm"])/100.  # This last 100. comes from normalization of 1./H0/OmegaM**0.5 meaning 1./wm/100.
		
		#if self.units=="Mpc":  # both yield the same result
		#	soundhorizon=s*1./self.cosmo_params["Hubble"]/np.sqrt(self.cosmo_params["OmegaM"])
		#if self.units=="h-1Mpc":
		#	soundhorizon=s*np.sqrt(1./self.cosmo_params["wm"])/100.  # This last 100. comes from normalization of 1./H0/OmegaM**0.5 meaning 1./wm/100.
		
		return soundhorizon
		
	def epoch_of_equality(self):
		self.zeq=(2.5e4)*self.cosmo_params["wm"]*(self.cosmo_params["Theta27"]**(-4))
		self.aeq=1./(1+self.zeq)
		
	def epoch_of_end_of_drag(self):
		b1=0.313*(self.cosmo_params["wm"]**(-0.419))*(1+0.607*(self.cosmo_params["wm"]**0.674) )
		b2=0.238*(self.cosmo_params["wm"]**0.223)
		self.zdrag=1291*( (self.cosmo_params["wm"]**0.251)*1./(1+0.659*(self.cosmo_params["wm"]**0.828)) )*(1+b1*(self.cosmo_params["wb"]**b2) )
		self.adrag=1./(1.+self.zdrag)
	
	def epoch_of_decoupling(self): 
		"""
		Equation (66) from Komatsu (2009), as proposed by Hu & Sugiyama (1996)
		"""
		g1=( 0.0783*(self.cosmo_params["wb"]**(-0.238))  )*1./( 1.+ 39.5*(self.cosmo_params["wb"]**(0.763)))
		g2=( 0.560 )*1./( 1.+21.1*(self.cosmo_params["wb"]**(0.181)))
		
		self.zdec=1048.*( 1+ 0.00124*(self.cosmo_params["wb"]**(-0.738)))*( 1.+ g1*self.cosmo_params["wm"]*g2)
		
	def speed_of_sound(self,z):
		# In units of the speed of light. Meaning, you need to multiply result by the speed of sound.
		R=self.baryon_to_photo_ratio_z(z)
		#print "R ", R,
		cs=1./(3*(1+R))**0.5
		#print "cs ", cs,
		return cs
		
	def baryon_to_photo_ratio_z(self,z):
		return 31.5*self.cosmo_params["wb"]*(1000./z)*(self.cosmo_params["Theta27"]**(-4))
	
class FittingParams:
	def __init__(self,verbose=False):
		self.verbose=verbose
		
	def set_fitting_params(self,params,k=None, pk=None):
		default_params={'kstar':0.15, 'Amc':3.} #,'bias':2.}
		cosmoparams=Cosmo_Params(params=params) # in case params is a string this will produce the empirical values, otherwise  cosmoparams.params and params should be the same
		params=cosmoparams.params
		
		if params != None:
			for key in default_params.keys(): # default parameters included
				if params.__contains__(key): 
					if self.verbose==True: print 'set_fitting_params, cosmology_parameters.py: Using INPUT parameter '+key+' value of  ',params[key]
				else:
					params[key]=default_params[key]
					if self.verbose==True: print 'set_fitting_params, cosmology_parameters.py: Using DEFAULT parameter '+key+' value of ',params[key]
		else:
			params= default_params
			if self.verbose==True: print "No parameters given - using default parameters"
			
		self.params=params
		"""
		self.manage_pk(k=k,pk=pk)
		
		self.set_bias_params()
		self.set_growth_params()
		self.set_kaiserdistortion_params()
		self.matter_params()
		"""
"""
class Growth_Params:
	def __init__(self,params=None):
		self.params=params
		if type(params) == str: self.empirical()
		self.assign_parameters()
	def empirical(self):
		if self.params[0:8] == "LasDamas": 
			print "cosmology_parameters.py: Using LasDamas input parameters"
			ld_params={'wm': 0.1225, 'wb': 0.0196,'OmegaK':0., 'Hubble':70.,'weq':-1.0,'ns':1.}
			if self.params == "LasDamas_0.16z0.44_Z":
				ld_params['redshift']=0.3145
			self.params=ld_params
"""
