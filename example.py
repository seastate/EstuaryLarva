import numpy as np
import matplotlib.pyplot as plt
from EstLarvSwim import Estuary, Larvae, SwimSim
plt.ion()
e = Estuary(infile='Estuaries/Willapa_medium.out')
L1 = Larvae(N=64,release_date=60,stage_durations_days=[4,10],stage_velocities=[-500e-6,500e-6],
              color='b',settle_in_substrate=1,substrate_extent=[-100,-40],
              release_depths=[0.1,1])
L2 = Larvae(N=64,release_date=77,stage_durations_days=[3,8],stage_velocities=[-250e-6,250e-6],
              color='r',settle_in_substrate=1,substrate_extent=[-80,-40],
              release_depths=[0.99,1])
S = SwimSim(estuary=e,larvae=[L1,L2],plot_interval=2*3600,days=[59.75,88])
S.run()
L = S.larvae[0]

