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

The complete builder performs four operations:

1. downloads the five immutable STiC inputs recorded in the packaged
   `sources.json` and verifies every SHA256 digest;
2. runs the pinned STiC/Wittmann EOS to regenerate the continuum and
   thermodynamic lookup covering 5000 A, HMI 6173 A, and Hinode 6301/6302 A;
3. extracts the HMI and Hinode absolute disk-centre references from the pinned
   FTS atlas and copies the reviewed atomic/instrument metadata;
4. seals the namespaced bundle and verifies its complete inventory and every
   digest against the committed production bundle.

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
