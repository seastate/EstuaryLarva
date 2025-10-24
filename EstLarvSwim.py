'''
Python code implementing a model of benthic marine invertebrate larval release,
transport and settlement in estuaries. Larvae are advected by estuarine flows
driven by freshwater inputs and salinity differences, and swim or sink vertically
according to specified temporal sequences, potentially including diel vertical
migration and stage-dependent swimming behaviors.

Estuarine flow is simulated using the unsteady eta2d model by Parker MacCready.
For details of the eta2d model, see:
MacCready, P. (2004). Toward a unified theory of tidally-averaged estuarine
salinity structure. Estuaries, 27, 561-570. doi:10.1007/BF02907644
MacCready, P. (2007). Estuarine Adjustment. Journal of Physical Oceanography,
 37, 2133-2145. doi:10.1175/JPO3082.1

The code structure is based on and generally follows MacCready's original matlab eta2d codes.

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from loadOctave import octaveStruct

small = 1.e-10

class SwimSim():
    """A class to facilitate executing a simulation of tidal estuarine flows and
       larval swimming and settlement behavior.
    """
    def __init__(self,estuary=None,larvae=[],days=[59.75,90],dt_sec=15*60,plot_interval=2*3600,
                 nbins=10):
        """Create a SwimSim instance.
        """
        self.estuary = estuary
        self.larvae = larvae
        self.days = days
        self.dt_sec = dt_sec
        self.next_plot = 0.
        self.plot_interval = plot_interval
        self.nbins = nbins
        # Initialize depths of larval cohorts
        for L in self.larvae:
            hs = estuary.bottom_profile(L.Xs)
            L.Zs = L.fractional_depths * hs
            #print(L.fractional_depths)
        # Initialize variables for the simulations
        self.t_start = self.days[0] * 24*60*60
        self.t_end = self.days[1] * 24*60*60
        self.t = self.t_start
        # Initialize times for stages of larval cohorts
        for L in self.larvae:
            L.stage_times = [L.release_time]
            L.stage_ages = [0.]
            for i,sd in enumerate(L.stage_durations):
                L.stage_times.append(L.stage_times[-1]+L.stage_durations[i])
                L.stage_ages.append(L.stage_ages[-1]+L.stage_durations[i])
        

    def run(self):
        """Execute a simulation using current parameters.
        """
        # Initialize the current time and flowfield
        self.t = self.t_start
        self.estuary.update_flow(t=self.t)
        # Main time loop
        #print('got here 1')
        while self.t <= self.t_end:
            self.t += self.dt_sec   # update current simulation time
            self.estuary.t = self.t # update the time in the estuary instance
            # If needed, update estuary flow
            #if self.t >= self.estuary.t_next_snapshot:
            #    print(f'updating estuary flow, t = {self.t},{self.estuary.t}')
            #    self.estuary.update_flow()
            #    print('got here 2')
            self.estuary.update_flow()
            #print('got here 3')
            # Update states of larvae that have been released
            for L in self.larvae:   # Cycle through larval cohorts, if released
                #print('got here 4')
                if L.release_time <= self.t:
                    # Update age and stage, for living non-settled larvae
                    L.age += self.dt_sec
                    L.stage = max([i for i,st in enumerate(L.stage_ages) if L.age>=st])
                    L.living = np.where(L.stages<L.nstages)
                    L.stages[L.living] = L.stage
                    # temporarily plot larvae
                    L.plot(ax=self.estuary.Cax)
                    # Update swimming velocities to current stages
                    L.swim_velocities = np.take(L.stage_velocities,L.stages)
                    # Get U velocites due to estuarine flow, using fractional depths to interpolate
                    # onto a regular grid
                    L.fractional_depths = L.Zs / self.estuary.bottom_profile(L.Xs)
                    L.fit_points = [self.estuary.zeta,self.estuary.xkm.flatten()]
                    # Generate interpolants for horizontal velocity U and eddy diffusivities
                    # in the horizontal (KH) and vertical (KS) directions
                    L.interpU = RegularGridInterpolator(L.fit_points, self.estuary.U)
                    L.interpKH = RegularGridInterpolator(L.fit_points, self.estuary.KH)
                    L.interpKS = RegularGridInterpolator(L.fit_points, self.estuary.KS)
                    L.larvae_points = np.array([-L.fractional_depths,L.Xs]).T
                    # Horizontal units are km so divide by 1000 after interpolating
                    L.Us = L.interpU(L.larvae_points,method='linear')/1000.
                    L.Ws = np.copy(L.swim_velocities)
                    L.KS = L.interpKS(L.larvae_points,method='linear')
                    L.KH = L.interpKH(L.larvae_points,method='linear')
                    # Calculate displacement of larvae (units are m for vertical, km for horizontal)
                    L.dXs = self.dt_sec*L.Us + 1e-3*np.random.normal(size=L.N)*np.sqrt(2*self.dt_sec*L.KH)
                    L.dZs = self.dt_sec*L.Ws + np.random.normal(size=L.N)*np.sqrt(2*self.dt_sec*L.KS)
                    # Set displacements to zero for larvae that are settled or "exported"
                    L.settled = np.where(L.stages==L.nstages+1)
                    L.dXs[L.settled] = 0.
                    L.dZs[L.settled] = 0.
                    L.exported = np.where(L.Xs==self.estuary.xkm.flatten()[-1])
                    L.dXs[L.exported] = 0.
                    L.dZs[L.exported] = 0.
                    # Update positions
                    L.Xs += L.dXs
                    L.Zs += L.dZs
                    # Keep larvae within the estuary
                    L.Xs[np.where(L.Xs>0.)] = 0.
                    L.Xs[np.where(L.Xs<self.estuary.xkm.flatten()[0])] = self.estuary.xkm.flatten()[0]
                    # Make sure no larvae can dig or fly
                    L.Zs[np.where(L.Zs<0.)] = 0.
                    L.Zs = np.minimum(L.Zs,self.estuary.bottom_profile(L.Xs))
                    # Allow eligible larvae to settle
                    if L.settle_in_substrate:
                        L.competent = np.where((L.stages==L.nstages-1) & (L.Zs==self.estuary.bottom_profile(L.Xs))
                                           & (L.Xs>L.substrate_extent[0]) & (L.Xs<=L.substrate_extent[1]))
                    else:
                        L.competent = np.where((L.stages==L.nstages-1) & (L.Zs==self.estuary.bottom_profile(L.Xs)))
                    L.stages[L.competent] = L.nstages+1
            if self.t >= self.next_plot:
                print(f'reached plotting interval..., t = {self.t}, {self.t/(24*60*60)}')
                self.next_plot = self.t + self.plot_interval
                self.plot()
                print('Larval stages:')
                print(self.larvae[0].stages)
                print('Larval velocities:')
                print(self.larvae[0].swim_velocities)
                #self.estuary.plot()
                #for L in self.larvae:   # Cycle through larval cohorts, if released
                #    if L.release_time <= self.t:
                #        L.plot(self.estuary.Cax,ma)

    def stats(self):
        """Collect some useful statistics about the current state of the estuary and larval cohorts.
        """
        # Collect histograms of vertical and horizontal distributions of living larvae
        self.hbins = np.linspace(self.estuary.xkm.flatten()[0],self.estuary.xkm.flatten()[-1],self.nbins)
        self.hcenters = 0.5 * (self.hbins[1:]+self.hbins[:-1])
        self.vbins = np.linspace(0,self.estuary.H.flatten().max(),self.nbins)
        self.vcenters = 0.5 * (self.vbins[1:]+self.vbins[:-1])
        # Plot horizontal and vertical distributions
        self.Hax = self.estuary.Efig.add_subplot(323)
        self.Hax.cla()
        self.Vax = self.estuary.Efig.add_subplot(324)
        self.Vax.cla()
        for L in self.larvae:   # Cycle through larval cohorts, if released
            if L.release_time <= self.t:
                L.histXs = np.histogram(L.Xs[np.where((L.stages!=L.nstages) &
                                                      (L.Xs<self.estuary.xkm.flatten()[-1]) &
                                                      (L.Xs>self.estuary.xkm.flatten()[0])) ],bins=self.hbins)
                L.histZs = np.histogram(L.Zs[np.where((L.stages!=L.nstages) &
                                                      (L.Xs<self.estuary.xkm.flatten()[-1]) &
                                                      (L.Xs>self.estuary.xkm.flatten()[0]))],bins=self.vbins)
                self.Hax.plot(self.hcenters,L.histXs[0],color=L.color)
                self.Vax.plot(L.histZs[0],self.vcenters,color=L.color)
        self.Hax.set_xlim(self.estuary.xkm.flatten()[0],self.estuary.xkm.flatten()[-1])
        self.Hax.set_ylim(0,25)
        self.Vax.set_ylim(0.,self.estuary.H.flatten().max())
        self.Vax.set_xlim(0,25)
        self.Vax.invert_yaxis()
        self.Vax.set_ylabel('Depth, $Z$ ($m$)')
        self.Vax.set_xlabel('Number individuals')
        self.Hax.set_xlabel('Streamwise position, $X$ ($km$)')
        self.Hax.set_ylabel('Number individuals')
               
    def plot(self):
        """Plot the current state of the estuary and larval cohorts.
        """
        self.estuary.plot()
        for i,L in enumerate(self.larvae):
            if L.release_time <= self.t:
                L.plot(self.estuary.Cax)
                #print(f'got here 6, cohort {i}')
                plt.pause(0.1)
        self.stats()
        plt.draw()
        #print('got here 5a')
        plt.pause(0.1)
        #print(f'got here 5b, times = {self.t}, {self.estuary.t}')

class Larvae():
    """A class to facilitate defining and simulating the fates of a larval
       population with specified characteristics.
    """
    def __init__(self,N=64,release_date=None,stage_durations_days=[],stage_velocities=[],
                 color='b',markers=['^','v','x','s'],
                 settle_in_substrate=True,substrate_extent=None,release_depths=None):
        """Create an Larvae instance, corresponding to a population of larvae with similar
           demographic and behavioral parameters.
        """
        # Record the demographic characteristics of this variant
        self.N = N
        self.age = 0.
        self.stage = 0
        # nstages is the number of planktonic stages, with distinct swimming parameters.
        # The sequence [0,nstages-1] corresponds to entries for stage duration and velocity.
        # Stage nstages+1 corresponds to settlement. Stage nstages corresponds to death.
        self.nstages = len(stage_durations_days)
        self.stage_durations_days = stage_durations_days 
        ## Settlement and death last forever with zero velocity; append these
        #print(stage_velocities)
        self.stage_velocities = stage_velocities + [0.,0.]
        #print(self.stage_velocities)
        #self.stage_velocities = stage_velocities  #.append([0.,0.]) 
        self.substrate_extent = substrate_extent
        self.settle_in_substrate = settle_in_substrate
        self.release_date = release_date
        self.release_depths = release_depths
        self.markers = markers
        self.color = color
        # Calculate cumulative stage durations, converting times from days to seconds
        self.stage_durations = [d*24*60*60 for d in self.stage_durations_days]
        
        self.release_time = self.release_date * 24*60*60
        # Initialize larval stage and swimming velocity
        self.stages = np.zeros([self.N],dtype='int64')  # larvae begin at stage 0
        self.swim_velocities = np.take(self.stage_velocities,self.stages)
        #self.swim_velocities = self.stage_velocites[0] * np.ones([1,self.N])
        # Initialize locations
        self.initial_positions = np.random.uniform(self.substrate_extent[0],self.substrate_extent[1],self.N)
        # Initialize fractional depths; absolute depths depend on estuary profiles so they
        # will be calculated as part of the simulation
        self.fractional_depths = np.random.uniform(self.release_depths[0],self.release_depths[1],self.N)
        # Placeholders for absolute horizontal and vertical positions
        self.Xs = np.copy(self.initial_positions)
        self.Zs = np.zeros([self.N])
        
    def plot(self,ax=None):
        """Plot larval position and stage on the given axis.
        """
        for i in range(self.nstages+2):  # Plot stages successively
            #indices = np.where(self.stages==i).flatten()
            indices = [j for j,s in enumerate(self.stages) if s == i]
            marker = self.markers[i]
            color = self.color
            ax.plot(self.Xs[indices],self.Zs[indices],color=color,marker=marker,linestyle='')


                               
class Estuary():
    """A class to facilitate defining and extracting properties of a specific
       estuary, such as geographical properties, flow characteristics as
       a function of location and time, and distributions and stages of
       larval populations.
    """
    def __init__(self,infile=None,verbose=False,readfile=True,larvae=[]):
        """Create an Estuary instance, with an option to load parameters from a specified
           input file.
        """
        self.input_data = None
        self.infile = infile
        self.larvae = larvae
        if infile and readfile:
            self.setup()
    
    def setup(self,infile=None,mixing=1,larvae=[],tiny = 1e-14):
        """Set up the flow scenario for an estuary using an eta2d input file.
        """
        self.mixing = mixing  # a switch to turn off turbulent mixing of larvae; probably unnecessary
        self.tiny = tiny      # a small number, generally used to remove rounding error and bit flipping
        if infile:  # update input file, if given
            self.infile = infile
        if len(larvae) > 0:  # add new larval variants, if given
            self.lavae.append(larvae)
        if not self.infile: # If input file isn't specified, prompt user
            print('Please specify estuary input file...')
            return
        # Load the fields in the eta2d input file, and ingest them as attributes
        self.input_data = octaveStruct(infile=self.infile)
        for key,value in self.input_data.ostr.items():
            setattr(self,key,value)
        # set up time variables
        self.td = self.T_vec/(24*60*60) # convert units from seconds to days
        self.tsd = self.Tsave_vec/(24*60*60)
        self.omega = 2*np.pi/(3600*12.42)
        #self.t = 0          # dummy values to force loading of flow information
        self.t = np.min(self.Tsave_vec_exact.flatten())
        self.next_time = -1  # old notation, probably to be replaced
        self.t_snapshot = -1.  # Current snapshot time, set at a dummy initial value
        self.i_snapshot = -1  # Index of the current snapshot time, set at a dummy initial value
        self.t_next_snapshot = -1 # For convenience in deciding when to update flow
        # set up plotting infrastructure
        self.Efig = plt.figure(figsize=(12,9))
        self.Efig.tight_layout()
        self.clevels = np.arange(-2,2,.1)



    def update_flow(self,t=None):
        """Update the flow scenario to correspond to time t. It is assumed that flow
           snapshots were saved with sufficient temporal resolution to use piecewise
           constant velocity fields for estimating larval transport. Therefore flow
           at time t is approximated by the most recent snapshot.
        """
        if t is not None:
            self.t = t
        # Use the latest snapshot previous to requested time
        self.i_snapshot = np.max(np.where(self.Tsave_vec_exact.flatten()<=self.t))
        self.t_snapshot = self.Tsave_vec_exact.flatten()[self.i_snapshot]
        if self.i_snapshot < self.Tsave_vec_exact.flatten().size-1:
            self.t_next_snapshot = self.Tsave_vec_exact.flatten()[self.i_snapshot+1]
        # Make some shortcuts for legibility (and in some cases, reshaping arrays)
        ii = self.i_snapshot
        nx = int(self.nx)
        ubar = np.reshape(self.ubar_mat[ii,:],[1,nx])
        ue = np.reshape(self.ue_mat[ii,:],[1,nx])
        sigma = np.reshape(self.Sigma[ii,:],[1,nx])
        sigmax = np.reshape(self.Sigmax[ii,:],[1,nx])
        Ks = np.reshape(self.Ks_mat[ii,:],[1,nx])
        Kh = np.reshape(self.Kh_mat[ii,:],[1,nx])
        x = np.reshape(self.x[0,:],[1,nx])    # copy, dropping spurious dimension
        xkm = np.reshape(self.xkm[0,:],[1,nx])    # copy, dropping spurious dimension
        lowsig = self.lowsig
        dx = self.dx
        H = self.H

        # derive bulk quantities
        ihi = int(np.min(np.where(sigma>=lowsig)))
        if ihi > 0:
            ilo = ihi-1
            xlo = np.interp(lowsig,sigma[ilo:ihi],x[ilo:ihi])
        else:
            xlo = x[0]

        x_new = np.linspace(xlo,x[-1],100)
        sigmax_new = np.interp(x_new,x.flatten(),sigmax.flatten())
        sigmax_mean = sigmax_new.mean()
        sigma_int = dx*np.trapezoid(sigma)

        # try smoothing things out in the tail
        sigmin = self.tiny
        tail_list = np.where(sigma<sigmin)
        sigma[tail_list] = 0
        sigmax[tail_list] = 0

        # set up to plot results
        # make the lower limit for plotting
        slist = np.where(sigma<=lowsig)[0]
        if slist.size > 0:
            nlow = max(slist)
        else:
            nlow = 0

        #nlow = np.max(np.where(sigma<=lowsig))
        if nlow > 9:
            xkm_low = xkm[nlow-10]
        else:
            xkm_low = min(xkm)

        xkm_hi = max(xkm)

        # calculate the diffusive fraction of up-estuary salt flux
        #    warning off MATLAB:divideByZero
        with np.errstate(divide='ignore'):
            nu = np.divide(np.multiply(self.Kh,sigmax),np.multiply(ubar,sigma))

        #nu = Kh.*sigmax ./ (ubar.*sigma);
        #    warning on MATLAB:divideByZero
        nu[0:nlow-1] = np.nan
        xx = np.linspace(xkm.flatten()[nlow],0,20)  # x-axis with fewer points
        nu = np.interp(xx,xkm.flatten()[nlow:-1],nu.flatten()[nlow:-1])

        # Calculate the streamfunction in x-zeta (psi)
        nz = 100
        zeta = np.linspace(-1,0,nz)   # dimensionless vertical coordinate
        # make the polynomial expressions  on the meshgrid
        X,Z = np.meshgrid(xkm,zeta)
        Z2 = Z * Z
        Z3 = Z2 * Z
        Z4 = Z3 * Z
        Z5 = Z4 * Z
        P1 = 0.5 - 1.5*Z2
        P2 = 1 - 9*Z2 - 8*Z3
        P3 = -(7/120) + 0.25*Z2 - 0.125*Z4
        P4 = -(1/12) + 0.5*Z2 -0.75*Z4 - 0.4*Z5
        UBAR = np.repeat(ubar,nz,axis=0)
        UE = np.repeat(ue,nz,axis=0)
        UP = UBAR*P1 + UE*P2
        #     U = UBAR + UP

        #   New lines from plot_tidal.m
        UT = np.repeat(self.Ut,nz,axis=0)

        # calculate the salinity field
        KS = np.repeat(Ks,nz,axis=0)
        KH = np.repeat(Kh,nz,axis=0)
        #     KV = np.ones([nz,1])*Kv
        H2 = np.repeat(H*H,nz,axis=0)
        SBAR = np.repeat(sigma*self.Socn,nz,axis=0)
        SBARX = np.repeat(sigmax*self.Socn,nz,axis=0)
        SP = H2*SBARX*(UBAR*P3 + UE*P4)/KS

        #     S = SBAR + SP
        # make a matrix of the actual depth
        ZZ = np.zeros([nz,nx])
        for iii in range(nx):
            ZZ[:,iii] = H.flatten()[iii]*zeta

        tsd = self.tsd.flatten()
        td = self.td.flatten()
        Qr_vec = self.Qr_vec.flatten()
        Ut0_vec = self.Ut0_vec.flatten()
        Socn_vec = self.Socn_vec.flatten()

        this_qr = np.interp(tsd[ii],td,Qr_vec)
        this_ut = np.interp(tsd[ii],td,Ut0_vec)
        this_so = np.interp(tsd[ii],td,Socn_vec)


        UTIDE = 1.5 * UT*(1-Z2) * np.cos(self.omega*self.t)
        U = UBAR + UP + UTIDE


        u_surface = U[-1,:]  # surface velocity (m s-1)

        #   New lines from plot_tidal.m
        STIDE = -1.5 * (1/self.omega) * SBARX*UT*(1-Z2) * np.sin(self.omega*self.t)
        S = SBAR + SP + STIDE

        S[np.where(S<0)] = 0

        # save variables that will be needed for transport calcs and plotting
        self.U = U
        self.S = S
        self.this_qr = this_qr
        self.this_ut = this_ut
        self.this_so = this_so
        self.X = X
        self.ZZ = ZZ
        self.Qr_vec = Qr_vec
        #self.UT = UT   # should this be UTIDE????
        self.Ut0_vec = Ut0_vec
        self.tsd = tsd
        self.zeta = zeta
        self.KS = KS
        self.KH = KH

    def bottom_profile(self,xs):
        """Calculate bottom depths at the horizontal positions in the array xs.
        """
        zs = np.interp(xs,self.xkm.flatten(),self.H.flatten())
        return zs
        
    def plot(self):
        """Plot the current flow and larval characteristics
        """
        # Create an axis to visualize the state of the estuary and the larval distribution    
        self.Efig.clf()
        self.Cax = self.Efig.add_subplot(313)
        #self.Cax.cla()

        self.Cax.set_facecolor('tan')
        
        CS = plt.contourf(self.X,-self.ZZ,self.U,self.clevels)
        plt.gca().invert_yaxis()
        
        #plt.title(f'Flow velocity, time = {self.t/(24*60*60):.2f}')
        plt.ylabel('Depth, $Z$ ($m$)')
        plt.xlabel('Streamwise position, $X$ ($km$)')

        CS.set_edgecolor('gray')
        print(CS.get_linewidth())
        CS.set_linewidth(0.01)
        #CB = plt.colorbar(CS, shrink=0.8, extend='both')
        self.Cax.clabel(CS, fontsize=7,colors='gray')
        #self.Cax.clabel(CS, CS.levels, fmt=fmt, fontsize=10)
  
        # Create an axis to display the tidal amplitude and riverine input flow rate
        self.Qax = self.Efig.add_subplot(311)
        self.Qax.cla()
        PLT_Q=self.Qax.plot(self.td.flatten(),self.Qr_vec/1000,'b')
        self.Qax.plot(self.t_snapshot/(24*60*60),self.this_qr/1000,'bo')
        self.Qax.set_xlabel('Time, $t$')
        self.Qax.set_ylabel('River flow ($Q_r/1000$)')
        self.Qax.set_ylim(0,2.5)
        self.Qax.yaxis.label.set_color('blue')
        self.Qax.spines['right'].set_color('blue')
        self.Qax.tick_params(axis='y', colors='blue')
        #Fax1.annotate(str(round(plot_F[-1],2)),xy=(t[-1],plot_F[-1]),xycoords='data',
        #                  xytext=(-10.,10.),textcoords='offset points')
        self.Qax2 = self.Qax.twinx()
        PLT_U=self.Qax2.plot(self.td.flatten(),self.Ut0_vec,'r')
        self.Qax2.plot(self.t_snapshot/(24*60*60),self.this_ut,'ro')
        self.Qax2.set_ylabel('Tidal amplitude ($U$)')
        self.Qax2.set_ylim(0,2.5)
        self.Qax2.yaxis.label.set_color('red')
        #ax.spines['bottom'].set_color('red')
        self.Qax2.spines['right'].set_color('red')
        self.Qax2.tick_params(axis='y', colors='red')

        # Set a title
        plt.suptitle(self.labtext[:-1],fontsize=14)
        plt.title(f'Flow velocity, time = {self.t/(24*60*60):.2f}')
        #plt.title(self.labtext[:-1])
        self.Efig.tight_layout()

        
    





            
'''            

        

    def setup_flow(self):
        """Set up the flow scenario.
        """
        # (Re)define these, in case larvae were added after the estuary was set up
        self.release_date = min([larva_type.release_date for larva_type in self.larvae])
        #t_index = max(find(tsd<release_date))-1
        self.t_index = max([tt for tt in self.tsd if tt<self.release_date])

        #self.t_index = -1
            #ii = self.t_index

            self.ubar = self.ubar_mat[self.t_index,:]
            self.ue = self.ue_mat[self.t_index,:]
            self.sigma = self.Sigma[self.t_index,:]
            self.sigmax = self.Sigmax[self.t_index,:]
            self.Ks = self.Ks_mat[self.t_index,:]
            sys = choose_an_estuary(n_system)
            %
            L= sys.L
            nx = sys.nx
            x = sys.x
            xkm = sys.xkm
            dx = sys.dx
            H = sys.H
            B = sys.B
            Qr = sys.Qr
            ubar = sys.ubar
            g = sys.g
            beta = sys.beta
            Socn = sys.Socn
            rho0 = sys.rho0
            Ut = sys.Ut
            Km = sys.Km
            Ks = sys.Ks
            Kh = sys.Kh
            Kv_calc = sys.Kv_calc
            Kh_calc = sys.Kh_calc
            n_system = sys.n_system
            labtext = sys.labtext
            A = sys.A
            c = sys.c
            lowsig = sys.lowsig
            clear sys

            ubar = ubar_mat[self.t_index,:]
            ue = ue_mat[self.t_index,:]
            sigma = Sigma[self.t_index,:]
            sigmax = Sigmax[self.t_index,:]
            Ks = Ks_mat[self.t_index,:]


            %   New lines from plot_tidal.m

            % derive bulk quantities
            ihi = find(sigma>=lowsig)
            ihi = ihi(1)
            if ihi > 1
                ilo = ihi-1
                if FreeMat_flag==1
                    xlo = interplin1(sigma(ilo:ihi),x(ilo:ihi),lowsig)
                else
                    xlo = interp1(sigma(ilo:ihi),x(ilo:ihi),lowsig)
                end
            else
                xlo = x(1)
            end
            x_new = linspace(xlo,x(end),100)
            if FreeMat_flag==1
                sigmax_new = interplin1(x,sigmax,x_new)
            else
                sigmax_new = interp1(x,sigmax,x_new)
            end
        #     phi_new = interplin1(x,phi,x_new)
            sigmax_mean = mean(sigmax_new)
        #     phi_mean = mean(phi_new)
            sigma_int = dx*trapz(sigma)

            # try smoothing things out in the tail
            sigmin = eps
            tail_list = find(sigma<sigmin)
            if ~isempty(tail_list)
                ntail = tail_list(end)
                sigma(1:ntail) = 0
                sigmax(1:ntail) = 0
            end

            # set up to plot results
            # make the lower limit for plotting
            nlow = max(find(sigma<=lowsig))
            if isempty(nlow) nlow = 1 end
            if nlow > 10
                xkm_low = xkm(nlow-10)
            else
                xkm_low = min(xkm)
            end
            xkm_hi = max(xkm)

            # calculate the diffusive fraction of up-estuary salt flux
            warning off MATLAB:divideByZero
            nu = Kh.*sigmax ./ (ubar.*sigma)
            warning on MATLAB:divideByZero
            nu(1:nlow) = NaN
            xx = linspace(xkm(nlow+1),0,20)
            # x-axis with fewer points
            if FreeMat_flag==1
                nu = interplin1(xkm(nlow+1:end),nu(nlow+1:end),xx)
            else
                nu = interp1(xkm(nlow+1:end),nu(nlow+1:end),xx)
            end


            # Calculate the streamfunction in x-zeta (psi)
            nz = 100
            zeta = linspace(-1,0,nz).transpose()   # dimensionless vertical coordinate
            # make the polynomial expressions  on the meshgrid
            [X,Z] = meshgrid(xkm,zeta)
            Z2 = Z.^2
            Z3 = Z.^3
            Z4 = Z.^4
            Z5 = Z.^5
            P1 = 0.5 - 1.5*Z2
            P2 = 1 - 9*Z2 - 8*Z3
            P3 = -(7/120) + 0.25*Z2 - 0.125*Z4
            P4 = -(1/12) + 0.5*Z2 -0.75*Z4 - 0.4*Z5
            UBAR = ones(nz,1)*ubar
            UE = ones(nz,1)*ue
            UP = UBAR.*P1 + UE.*P2
        #     U = UBAR + UP


            #   New lines from plot_tidal.m

            UT = ones(nz,1) * Ut

            # calculate the salinity field
            KS = ones(nz,1)*Ks
            KH = ones(nz,1)*Kh
        #     KV = ones(nz,1)*Kv
            H2 = ones(nz,1)*(H.*H)
            SBAR = ones(nz,1)*sigma*Socn
            SBARX = ones(nz,1)*sigmax*Socn
            SP = H2.*SBARX.*(UBAR.*P3 + UE.*P4)./KS
        #     S = SBAR + SP

            # make a matrix of the actual depth
            for iii = 1:nx
                ZZ(:,iii) = H(iii)*zeta
            end



            if FreeMat_flag==1
                this_qr = interplin1(td,Qr_vec,tsd(ii))
                this_ut = interplin1(td,Ut0_vec,tsd(ii))
                this_so = interplin1(td,Socn_vec,tsd(ii))

            else
                this_qr = interp1(td,Qr_vec,tsd(ii))
                this_ut = interp1(td,Ut0_vec,tsd(ii))
                this_so = interp1(td,Socn_vec,tsd(ii))
            end


        end
    
UTIDE = 1.5 * UT .* (1-Z2) * cos(omega*t)
U = UBAR + UP + UTIDE


u_surface = U(end,:)   # surface velocity (m s-1)
#

#   New lines from plot_tidal.m
STIDE = -1.5 * (1/omega) * SBARX .* UT .* (1-Z2) * sin(omega*t)
S = SBAR + SP + STIDE

S(S<0) = 0


'''
        



#def choose_an_estuary():











"""
        X_substrate = [-40 -20
                        -40   -20
                        ]     #   Horizontal position from which the particles are released


        Z_substrate = [.79 .8
                       .79 .8
                        ]        #   Range of depths from which particles are released (as fractions of depth)
                




"""


