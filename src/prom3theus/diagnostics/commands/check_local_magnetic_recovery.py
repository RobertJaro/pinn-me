"""Check line-forming magnetic fields and formal depth convergence on real pixels.

Select pixels by observed polarization, never by a magnetic-field target.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from prom3theus.rt.ray_geometry import sample_shell_characteristic
from prom3theus.observations.loading import buffered_dataloader
from prom3theus.muram import config as _register_muram_schema
from prom3theus.config import load_config
from prom3theus.application.joint_assembly import build_joint_runtime
from prom3theus.validation.reference_transfer import PolarizedFormalSolver, RayDistancePath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration', type=Path)
    parser.add_argument('--pixels', type=int, default=3)
    parser.add_argument('--depth', type=int, default=513)
    parser.add_argument('--microturbulence-factor',type=float,default=1.,
                        help='Read-only formal-RT sensitivity test; leaves checkpoint and model untouched')
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.configuration)
    runtime = build_joint_runtime(config)
    checkpoint = torch.load(config.solver.output_directory/'last.ckpt', map_location='cpu', weights_only=False)
    runtime.model.load_state_dict({k.removeprefix('model.'): v for k,v in checkpoint['state_dict'].items() if k.startswith('model.')})
    runtime.model.eval()
    term = runtime.model.terms['photospheric_stokes']
    window = term.synthesis_window
    if args.microturbulence_factor != 1:
        original_atmosphere=window.evaluate_atmosphere
        def counterfactual_atmosphere(position,time,**kwargs):
            state=dict(original_atmosphere(position,time,**kwargs))
            state['microturbulence']=state['microturbulence']*args.microturbulence_factor
            return state
        window.evaluate_atmosphere=counterfactual_atmosphere
    data = runtime.streams['photospheric_stokes'].data_module
    dataset = data.evaluation_dataset()
    collate = data.evaluation_collate_fn()
    strengths = []
    with torch.no_grad():
        for batch in buffered_dataloader(dataset, batch_size=128, num_workers=data.num_workers, collate_fn=collate):
            _, target = term.objective.comparison_profiles(batch['stokes'], batch['stokes'])
            strengths.extend(target[:,1:].square().mean((1,2)).tolist())
    indices = np.argsort(strengths)[-args.pixels:][::-1].copy()
    output = config.solver.output_directory/f'magnetic_validation_{checkpoint["global_step"]:06d}'
    if args.microturbulence_factor != 1:
        output=output.with_name(output.name+f'_micro_factor_{args.microturbulence_factor:g}')
    output.mkdir(exist_ok=True)
    reports = []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(args.pixels,5,figsize=(18,3*args.pixels),squeeze=False,layout='constrained')
    for row,index in enumerate(indices):
        batch = collate([dataset[int(index)]])
        with torch.no_grad():
            geo = window.boundary_queries(batch['coordinates'],batch['ray_direction'])
            n, time = geo['direction'], geo['time_hours']
            distance = torch.linspace(0,1,args.depth,dtype=torch.float64)*geo['path_length_m'][0]
            chart, local_height, position = sample_shell_characteristic(
                geo, distance[None], solar_radius_m=window.solar_radius_m)
            state = window.evaluate_atmosphere(position[0],time.expand(args.depth),
                chart=chart[0],height=local_height[0])
            height = local_height[0]/1e6
            alpha = window.coefficients.plasma_provider.volume_extinction_at_5000(state['temperature'],state['gas_pressure'])
            dtau = .5*(alpha[1:]+alpha[:-1])*(distance[1:]-distance[:-1])
            tau = torch.cat((dtau.new_zeros(1),dtau.cumsum(0)))
            b = state['magnetic_field'].norm(dim=-1)
            query = term.spectral_queries(batch)
            # Explicit double precision reconstruction for this reference check.
            wave = (query.origin_angstrom + query.offset_angstrom.double()).reshape(-1)
            matrices,sources = [],[]
            for start in range(0,args.depth,8):
                end = min(start+8,args.depth)
                local = window.coefficients({k:v[start:end] for k,v in state.items()},wave,direction=n.expand(end-start,3),stokes_basis=window.basis(n).expand(end-start,3,3),radiance_scale=window.radiance_scale)
                matrices.append(local.propagation_m1); sources.append(local.source)
            matrix,source = torch.cat(matrices),torch.cat(sources)
            incoming = None
            if window.configuration['incoming_temperature_k'] is not None:
                from prom3theus.rt.opacity import planck_lambda
                temperature = source.new_tensor(window.configuration['incoming_temperature_k'])
                intensity = planck_lambda(temperature, wave, radiance_scale=window.radiance_scale)
                zero = torch.zeros_like(intensity)
                incoming = torch.stack((intensity, zero, zero, zero), dim=-1).double()
            def formal(stride):
                return PolarizedFormalSolver()(matrix[::stride].double(),source[::stride].double(),torch.arange(len(distance[::stride]),dtype=torch.float64),path=RayDistancePath(distance[::stride]),bottom_boundary=incoming).T[None].float()
            raw, coarse = formal(1),formal(2)
            production = term.predict(batch)
            # Reuse the production detector rotation, normalization and PSF.
            reference = term.observe_stokes(batch, raw, direction=n)
            p,target = term.objective.comparison_profiles(production,batch['stokes'])
            f,_ = term.objective.comparison_profiles(reference,batch['stokes'])
            selected = (tau>=.01)&(tau<=1)
            def magnetic_at_tau(q):
                if q > float(tau[-1]):
                    return None
                j=int(torch.searchsorted(tau,tau.new_tensor(q)).clamp(1,len(tau)-1))
                fraction=((q-tau[j-1])/(tau[j]-tau[j-1]).clamp_min(1e-30)).to(state['magnetic_field'])
                return float(torch.lerp(state['magnetic_field'][j-1],state['magnetic_field'][j],fraction).norm())
            report = dict(index=int(index), B_tau_001_to_1_gauss=b[selected].tolist(), bottom_tau500=float(tau[-1]),
                          B_at_tau_gauss={str(q):magnetic_at_tau(q) for q in (.01,.1,1)},
                          production_rms=dict(zip('IQUV',(p-target).square().mean((0,2)).sqrt().tolist())),
                          formal_rms=dict(zip('IQUV',(f-target).square().mean((0,2)).sqrt().tolist())),
                          formal_refinement_max=float((raw-coarse).abs().max()))
            reports.append(report)
            np.savez_compressed(output/f'pixel_{index}.npz',height_megameter=height.numpy(),tau500=tau.numpy(),magnetic_gauss=state['magnetic_field'].numpy(),temperature=state['temperature'].numpy(),pressure=state['gas_pressure'].numpy(),production=p.numpy(),formal=f.numpy(),target=target.numpy())
            for comp in range(4):
                ax=axes[row,comp]; x=term.wavelength_angstrom.numpy()
                formal_label=('Atmosphere: formal RT' if args.microturbulence_factor == 1
                              else f'Formal sensitivity: micro × {args.microturbulence_factor:g}')
                ax.plot(x,target[0,comp],color='black',label='Observed');ax.plot(x,p[0,comp],label='model');ax.plot(x,f[0,comp],'--',label=formal_label)
                ax.set_title(f'Pixel {index}: {"IQUV"[comp]}');ax.set_xlabel('Air wavelength [Å]')
                ax.set_ylabel(f'{"IQUV"[comp]} / observed continuum')
            axes[row,4].semilogx(tau.clamp_min(1e-8),b);axes[row,4].axhline(1000,color='grey',ls=':');axes[row,4].set(xlim=(1e-4,10),xlabel='Continuum tau500',ylabel='|B| [G]')
    axes[0,0].legend(fontsize=7)
    fig.savefig(output/'comparison.png',dpi=130);plt.close(fig)
    result=dict(step=checkpoint['global_step'],pixels=reports,microturbulence_factor=args.microturbulence_factor)
    if len(dataset) <= 256:
        maps = []
        with torch.no_grad():
            for batch in buffered_dataloader(dataset, batch_size=16, num_workers=data.num_workers, collate_fn=collate):
                geo=window.boundary_queries(batch['coordinates'],batch['ray_direction'])
                count=len(geo['direction']); depth=513
                distance=geo['path_length_m'][:,None]*torch.linspace(0,1,depth,dtype=torch.float64)
                chart,local_height,position=sample_shell_characteristic(
                    geo,distance,solar_radius_m=window.solar_radius_m)
                time=geo['time_hours'][:,None].expand(count,depth)
                state=window.evaluate_atmosphere(position.reshape(-1,3),time.reshape(-1),
                    chart=chart.reshape(-1,2),height=local_height.reshape(-1))
                alpha=window.coefficients.plasma_provider.volume_extinction_at_5000(state['temperature'],state['gas_pressure']).reshape(count,depth)
                tau=torch.cat((alpha.new_zeros(count,1),(.5*(alpha[:,1:]+alpha[:,:-1])*(distance[:,1:]-distance[:,:-1])).cumsum(1)),dim=1)
                index=torch.searchsorted(tau.contiguous(),tau.new_full((count,1),.1)).clamp(1,depth-1)[:,0]
                row=torch.arange(count); lower=tau[row,index-1]; upper=tau[row,index]
                fraction=((.1-lower)/(upper-lower).clamp_min(1e-30)).float()
                def at_tau(name):
                    values=state[name].reshape(count,depth,-1)
                    return torch.lerp(values[row,index-1],values[row,index],fraction[:,None])
                magnetic=at_tau('magnetic_field')
                radial=position[row,index]/position[row,index].norm(dim=-1,keepdim=True)
                velocity=(at_tau('velocity_field')*radial).sum(-1)/1000
                values=torch.stack((magnetic.norm(dim=-1),(magnetic*geo['direction']).sum(-1),velocity,at_tau('temperature')[:,0]),-1)
                values[tau[:,-1]<.1]=torch.nan
                maps.append(torch.cat((batch['coordinates'][:,:2],values),-1).numpy())
        mapped=np.concatenate(maps)
        np.savez_compressed(output/'tau01_map.npz',values=mapped,columns=['x_megameter','y_megameter','B_gauss','B_los_gauss','radial_velocity_km_s','temperature_k'])
        fig,axes=plt.subplots(1,4,figsize=(15,4),layout='constrained')
        for j,(label,cmap) in enumerate(zip(('|B| [G]','B along ray [G]','Radial velocity [km/s]','Temperature [K]'),('magma','RdBu_r','RdBu_r','inferno'))):
            artist=axes[j].scatter(mapped[:,0],mapped[:,1],c=mapped[:,j+2],s=200,marker='s',cmap=cmap)
            fig.colorbar(artist,ax=axes[j]);axes[j].set(title=label,xlabel='x [megameter]',ylabel='y [megameter]',aspect='equal')
        fig.suptitle(f'Learned atmosphere at continuum tau500=0.1; step {checkpoint["global_step"]}')
        fig.savefig(output/'tau01_map.png',dpi=140);plt.close(fig)
        valid=np.isfinite(mapped[:,2])
        result['map_tau01']={'pixel_count':len(mapped),'valid_fraction':float(valid.mean()),'B_min_gauss':float(np.nanmin(mapped[:,2])),
                             'B_median_gauss':float(np.nanmedian(mapped[:,2])),'B_max_gauss':float(np.nanmax(mapped[:,2])),
                             'fraction_above_1kG':float(np.mean(mapped[valid,2]>=1000))}
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
