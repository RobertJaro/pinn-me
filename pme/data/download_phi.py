import sunpy_soar
from sunpy.net import Fido, attrs as a
from astropy.time import Time

t_start = Time('2024-03-23T22:00:00', format='isot', scale='utc')
t_end = Time('2024-03-24T03:00:00', format='isot', scale='utc')

results_phi = Fido.search(a.Instrument('PHI'), a.Time(t_start.value, t_end.value), a.soar.Product('phi-hrt-stokes'))
files_phi = Fido.fetch(results_phi, path='/glade/work/rjarolim/data/phi_stokes/2024_03_23')

# binc
results_phi = Fido.search(a.Instrument('PHI'), a.Time(t_start.value, t_end.value), a.soar.Product('phi-hrt-binc'))
Fido.fetch(results_phi, path='/glade/work/rjarolim/data/phi_stokes/2024_03_23')

# bazi
results_phi = Fido.search(a.Instrument('PHI'), a.Time(t_start.value, t_end.value), a.soar.Product('phi-hrt-bazi'))
Fido.fetch(results_phi, path='/glade/work/rjarolim/data/phi_stokes/2024_03_23')

# bmag
results_phi = Fido.search(a.Instrument('PHI'), a.Time(t_start.value, t_end.value), a.soar.Product('phi-hrt-bmag'))
Fido.fetch(results_phi, path='/glade/work/rjarolim/data/phi_stokes/2024_03_23')

# vlos
results_phi = Fido.search(a.Instrument('PHI'), a.Time(t_start.value, t_end.value), a.soar.Product('phi-hrt-vlos'))
Fido.fetch(results_phi, path='/glade/work/rjarolim/data/phi_stokes/2024_03_23')



t_start = Time('2024-03-29T00:00:00', format='isot', scale='utc')
t_end = Time('2024-03-30T00:00:00', format='isot', scale='utc')
results_phi = Fido.search(a.Instrument('PHI'), a.Time(t_start.value, t_end.value), a.soar.Product('phi-fdt-blos'), a.Level('L2'))
Fido.fetch(results_phi, path='/glade/work/rjarolim/data/phi_stokes/fdt_2024_03_29')
results_phi = Fido.search(a.Instrument('PHI'), a.Time(t_start.value, t_end.value), a.soar.Product('phi-fdt-bmag'), a.Level('L2'))
Fido.fetch(results_phi, path='/glade/work/rjarolim/data/phi_stokes/fdt_2024_03_29')