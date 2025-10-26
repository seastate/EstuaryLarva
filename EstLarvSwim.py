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
        # Create a graphics window and add axes for separate plots
        self.Efig = plt.figure(figsize=(12,9))
        self.Efig.tight_layout()
        self.Qax = self.Efig.add_subplot(311)
        self.Qax2 = self.Qax.twinx()
        self.Cax = self.Efig.add_subplot(313)
        self.Hax = self.Efig.add_subplot(323)
        self.Vax = self.Efig.add_subplot(324)

    def run(self):
        """Execute a simulation using current parameters.
        """
        # Initialize the current time and flowfield
        self.t = self.t_start
        self.estuary.update_flow(t=self.t)
        # Main time loop
        while self.t <= self.t_end:
            self.t += self.dt_sec   # update current simulation time
            self.estuary.t = self.t # update the time in the estuary instance
            self.estuary.update_flow()
            for L in self.larvae:   # Cycle through larval cohorts, if released
                if L.release_time <= self.t:
                    # Update age and stage, for living non-settled larvae
                    L.age += self.dt_sec
                    L.stage = max([i for i,st in enumerate(L.stage_ages) if L.age>=st])
                    L.living = np.where(L.stages<L.nstages)
                    L.stages[L.living] = L.stage
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
                print(f'Plotting interval..., t = {self.t}, {self.t/(24*60*60)}')
                self.next_plot = self.t + self.plot_interval
                self.plot()

    def stats(self,Hax=None,Vax=None):
        """Collect some useful statistics about the current state of the estuary and larval cohorts.
        """
        # Collect histograms of vertical and horizontal distributions of living larvae
        self.hbins = np.linspace(self.estuary.xkm.flatten()[0],self.estuary.xkm.flatten()[-1],self.nbins)
        self.hcenters = 0.5 * (self.hbins[1:]+self.hbins[:-1])
        self.vbins = np.linspace(0,self.estuary.H.flatten().max(),self.nbins)
        self.vcenters = 0.5 * (self.vbins[1:]+self.vbins[:-1])
        for L in self.larvae:   # Cycle through larval cohorts, if released
            if L.release_time <= self.t:
                L.histXs = np.histogram(L.Xs[np.where((L.stages!=L.nstages) &
                                                      (L.Xs<self.estuary.xkm.flatten()[-1]) &
                                                      (L.Xs>self.estuary.xkm.flatten()[0])) ],bins=self.hbins)
                L.histZs = np.histogram(L.Zs[np.where((L.stages!=L.nstages) &
                                                      (L.Xs<self.estuary.xkm.flatten()[-1]) &
                                                      (L.Xs>self.estuary.xkm.flatten()[0]))],bins=self.vbins)
        # If axes are supplied, plot the horizontal and vertical distributions
        if Hax is not None:
            self.Hax.cla()
            for L in self.larvae:   # Cycle through larval cohorts, if released
                if L.release_time <= self.t:
                    self.Hax.plot(self.hcenters,L.histXs[0],color=L.color)
            self.Hax.set_xlim(self.estuary.xkm.flatten()[0],self.estuary.xkm.flatten()[-1])
            self.Hax.set_ylim(0,25)
            self.Hax.set_xlabel('Streamwise position, $X$ ($km$)')
            self.Hax.set_ylabel('Number individuals')
        if Vax is not None:
            self.Vax.cla()
            for L in self.larvae:   # Cycle through larval cohorts, if released
                if L.release_time <= self.t:
                    self.Vax.plot(L.histZs[0],self.vcenters,color=L.color)
            self.Vax.set_ylim(0.,self.estuary.H.flatten().max())
            self.Vax.set_xlim(0,25)
            self.Vax.invert_yaxis()
            self.Vax.set_ylabel('Depth, $Z$ ($m$)')
            self.Vax.set_xlabel('Number individuals')
               
    def plot(self):
        """Plot the current state of the estuary and larval cohorts.
        """
        self.Efig.suptitle(self.estuary.labtext[:-1],fontsize=14)
        self.estuary.plot(Cax=self.Cax,Qax=self.Qax,Qax2=self.Qax2)
        for i,L in enumerate(self.larvae):
            if L.release_time <= self.t:
                L.plot(Cax=self.Cax)
        self.stats(Hax=self.Hax,Vax=self.Vax)
        self.Efig.tight_layout()
        plt.draw()
        plt.pause(0.1)


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
        # Settlement and death last forever with zero velocity; append these
        self.stage_velocities = stage_velocities + [0.,0.]
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
        # Initialize locations
        self.initial_positions = np.random.uniform(self.substrate_extent[0],self.substrate_extent[1],self.N)
        # Initialize fractional depths; absolute depths depend on estuary profiles so they
        # will be calculated as part of the simulation
        self.fractional_depths = np.random.uniform(self.release_depths[0],self.release_depths[1],self.N)
        # Placeholders for absolute horizontal and vertical positions
        self.Xs = np.copy(self.initial_positions)
        self.Zs = np.zeros([self.N])
        
    def plot(self,Cax=None):
        """Plot larval position and stage on the given axis.
        """
        for i in range(self.nstages+2):  # Plot stages successively
            #indices = np.where(self.stages==i).flatten()
            indices = [j for j,s in enumerate(self.stages) if s == i]
            marker = self.markers[i]
            color = self.color
            Cax.plot(self.Xs[indices],self.Zs[indices],color=color,marker=marker,linestyle='')

                               
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
    
    def setup(self,infile=None,mixing=1,larvae=[],tiny = 1e-14,Efig=None,clevels=np.arange(-2,2,.1)):
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
        self.t = np.min(self.Tsave_vec_exact.flatten())
        self.next_time = -1  # old notation, probably to be replaced
        self.t_snapshot = -1.  # Current snapshot time, set at a dummy initial value
        self.i_snapshot = -1  # Index of the current snapshot time, set at a dummy initial value
        self.t_next_snapshot = -1 # For convenience in deciding when to update flow
        # Set levels for velocity contours
        self.clevels = clevels

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
        if nlow > 9:
            xkm_low = xkm[nlow-10]
        else:
            xkm_low = min(xkm)
        xkm_hi = max(xkm)
        # calculate the diffusive fraction of up-estuary salt flux
        #    suppress warning of divideByZero
        with np.errstate(divide='ignore'):
            nu = np.divide(np.multiply(self.Kh,sigmax),np.multiply(ubar,sigma))
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
        #   New lines from plot_tidal.m
        UT = np.repeat(self.Ut,nz,axis=0)
        # calculate the salinity field
        KS = np.repeat(Ks,nz,axis=0)
        KH = np.repeat(Kh,nz,axis=0)
        H2 = np.repeat(H*H,nz,axis=0)
        SBAR = np.repeat(sigma*self.Socn,nz,axis=0)
        SBARX = np.repeat(sigmax*self.Socn,nz,axis=0)
        SP = H2*SBARX*(UBAR*P3 + UE*P4)/KS
        # make a matrix of the actual depth
        ZZ = np.zeros([nz,nx])
        for iii in range(nx):
            ZZ[:,iii] = H.flatten()[iii]*zeta
        tsd = self.tsd.flatten()
        td = self.td.flatten()
        Qr_vec = self.Qr_vec.flatten()
        Ut0_vec = self.Ut0_vec.flatten()
        Socn_vec = self.Socn_vec.flatten()
        # Save current quantities for plotting
        this_qr = np.interp(tsd[ii],td,Qr_vec)
        this_ut = np.interp(tsd[ii],td,Ut0_vec)
        this_so = np.interp(tsd[ii],td,Socn_vec)
        #
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
        
    def plot(self,Qax=None,Qax2=None,Cax=None):
        """Plot the current flow. Cax is an axis for a countour plot of velocity.
           Qax is an axis for riverine input flow rate. Qax2 is an axis for tidal
           amplitude. Each of these is plotted if the axis is supplied. The
           expectation is that Qax2 is a twinx of Qax but that is not required.
        """
        if Cax is not None:
            Cax.cla()
            Cax.set_facecolor('tan')  # Color space beneath the water like mud
            CS = Cax.contourf(self.X,-self.ZZ,self.U,self.clevels)
            Cax.invert_yaxis()
            Cax.set_ylabel('Depth, $Z$ ($m$)')
            Cax.set_xlabel('Streamwise position, $X$ ($km$)')
            # Tweak cotours to be visible but not distracting
            CS.set_edgecolor('gray')
            print(CS.get_linewidth())
            CS.set_linewidth(0.01)
            Cax.clabel(CS, fontsize=7,colors='gray')
        # Display the tidal amplitude and riverine input flow rate
        if Qax is not None:
            Qax.cla()
            PLT_Q=Qax.plot(self.td.flatten(),self.Qr_vec/1000,'b')
            Qax.plot(self.t_snapshot/(24*60*60),self.this_qr/1000,'bo')
            Qax.set_xlabel('Time, $t$')
            Qax.set_ylabel('River flow ($Q_r/1000$)')
            Qax.set_ylim(0,2.5)
            Qax.yaxis.label.set_color('blue')
            Qax.spines['right'].set_color('blue')
            Qax.tick_params(axis='y', colors='blue')
            Qax.set_title(f'Flow velocity, time = {self.t/(24*60*60):.2f}')
        if Qax2 is not None:
            Qax2.cla()
            PLT_U=Qax2.plot(self.td.flatten(),self.Ut0_vec,'r')
            Qax2.plot(self.t_snapshot/(24*60*60),self.this_ut,'ro')
            Qax2.set_ylabel('Tidal amplitude ($U$)')
            Qax2.yaxis.set_label_position('right')
            Qax2.set_ylim(0,2.5)
            Qax2.yaxis.label.set_color('red')
            Qax2.spines['right'].set_color('red')
            Qax2.tick_params(axis='y', colors='red')



        
