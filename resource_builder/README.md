# LTE resource builder

This flat directory is the offline scientific-resource build system. It is kept in
the repository for reproducibility, but it is deliberately outside
`src/prom3theus`, is not installed in wheels, and is never imported by an
inversion or training run.

Resource ownership is split into peer modules:

- `common_atomic.py` builds the common atomic metadata and STiC/Wittmann table;
- `hinode_sp.py` builds the complete Hinode/SP namespace;
- `hmi_stokes.py` builds the complete HMI namespace;
- `build.py` verifies pinned inputs, invokes each registered builder, seals the
  complete bundle, and checks byte-for-byte reproducibility.

Adding another instrument means adding one new peer module with a `build()`
function and registering it in `build.py`. Instrument generation logic must not
be added to the runtime package or the common atomic builder.

The complete builder performs five operations:

1. downloads the five immutable STiC inputs and the pinned CHIANTI ionization-
   equilibrium/version inputs recorded in `sources.json`, then verifies every
   SHA256 digest;
2. runs the pinned STiC/Wittmann EOS to regenerate the continuum and
   thermodynamic lookup covering 5000 A, HMI 6173 A, and Hinode 6301/6302 A,
   and generates the complete pinned FALC_82 reference atmosphere;
3. reduces the default CHIANTI 11.0.2 zero-density coronal equilibrium to the
   single required free-electrons-per-H mapping using the same STiC abundance
   and atomic-mass convention;
4. extracts the HMI and Hinode absolute disk-centre references from the pinned
   FTS atlas and copies the reviewed atomic/instrument metadata;
5. seals the namespaced bundle and verifies its complete inventory and every
   digest against the committed production bundle.

The compact CHIANTI thermodynamic table contains no emissivities, rates, or ion
fractions. It stores only `log10(electrons/H)`, its required PCHIP tangents, and
the STiC-mixture constants needed to derive density, electron density, and mean
particle mass from one shared hydrogen density. Runtime uses no CHIANTI
dependency. It preserves STiC through 10 kK and joins to this mapping through
31.6 kK using a bounded electron logit and a shared non-electron particle count.
It then keeps native CHIANTI exact through 10 MK and transitions to the fully
ionized analytic limit with a C1 log-temperature blend over exactly 10--20 MK;
the analytic limit is used at and above 20 MK. The complete native
CHIANTI 4--9 log-temperature grid is retained so shape-preserving endpoint
slopes do not depend on a cropped table. The runtime contract also records the
shared STiC boundary-cell cubic and bounded pressure-continuation convention,
including its C1 tangent-decay shoulder.

`common/falc_reference_atmosphere.json` preserves two coordinated views of the
same pinned STiC FALC_82 atmosphere. The 25-point line-formation view on
`-5 <= log10(tau500) <= 1` is unchanged, so the existing optical-depth-to-height
ray mapping is stable. The physical-height view restores all 37 native samples
with `log10(tau500) < -5`; its true top is 2.073502459 Mm at 100 kK and
0.0319344213 Pa. Runtime begins the coronal continuation only at this native
top: a quintic smootherstep in `log(T)` reaches the configured 1 MK at 2.5 Mm
and is isothermal above it, while pressure is integrated hydrostatically with
inverse-square gravity and `HybridSolarEOS`. The builder stores the pinned FALC
values and their provenance; the runtime constructs the configurable coronal
continuation rather than embedding it in the resource.

Create an isolated Python 3.11 environment, install the tested generator
dependencies, then run from the project root:

```bash
python -m pip install -r resource_builder/requirements.txt
python -m resource_builder.build \
  --output-directory build/reproduced-lte-resources
```

The output directory must not already contain files. Downloads are cached in
`~/.cache/prom3theus/resource-sources-v1` by default. The command exits with an
error if a source checksum, generated checksum, reviewed checksum, or final
inventory differs from the committed production contract.

To install a verified reproduction, compare/review it first and then replace
`src/prom3theus/resources/data` as one complete directory. Do not overlay
individual files: the runtime intentionally rejects stale or mixed bundles.
