import sunpy_soar
from sunpy.net import Fido, attrs as a
from astropy.time import Time

t_start = Time('2024-03-23T00:00', format='isot', scale='utc')
t_end = Time('2024-03-25T00:00', format='isot', scale='utc')

results_phi = Fido.search(a.Instrument('PHI'), a.Time(t_start.value, t_end.value), a.soar.Product('phi-hrt-stokes'), a.soar.SOOP('L_BOTH_HRES_HCAD_Major-Flare'))

files_phi = Fido.fetch(results_phi, path='/glade/work/rjarolim/data/phi_stokes/2024_03_flare')


