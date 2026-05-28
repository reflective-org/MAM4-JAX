# MAM4-JAX

A JAX port of the **MAM4 (Modal Aerosol Module, 4-mode) aerosol-microphysics box model**, validated against the E3SM Fortran reference.

The goal is a modern, readable, reproducible implementation of MAM4's microphysical processes — calcsize, water uptake, gas–aerosol exchange, nucleation, coagulation, and rename — written in JAX with `float64` precision and verified element-wise against the Fortran reference to a relative error of `1e-6`.

> **Status:** scaffolding. The Fortran reference has been imported; the JAX package is not yet started. See `CLAUDE.md` for working rules and architecture notes that govern the port.

## Repository layout

```
mam4-jax/
├── CLAUDE.md                 # Working rules, architecture, validation workflow
├── README.md                 # (this file)
└── mam4-original-src-code/   # Vendored Fortran reference — read-only
```

The JAX package, test harness, validation data, and design docs (`ARCHITECTURE.md`, `PROGRESS.md`, `PLANS.md`, `KEY_DECISIONS.md`, `DEFERRED.md`, `FEATURES.md`) will land as the port progresses.

## Fortran reference: provenance

The contents of `mam4-original-src-code/` are a **frozen snapshot** vendored into this repository. They are not a submodule and will not auto-update from upstream.

| Field | Value |
| --- | --- |
| Snapshot taken | 2025-12-10 |
| Immediate source | https://github.com/reflective-org/MAM4_box_model |
| Commit | [`4150e2d582a8fa6debca5009d22ec000496bd405`](https://github.com/reflective-org/MAM4_box_model/commit/4150e2d582a8fa6debca5009d22ec000496bd405) |
| Upstream project | https://github.com/kaizhangpnl/MAM_box_model (Kai Zhang et al., PNNL) |

If you need to inspect the Fortran reference's history, diff against later upstream versions, or reproduce the snapshot, clone the source above at the pinned commit.

The vendored snapshot is **read-only** within this repo — do not modify the files under `mam4-original-src-code/`. To update the snapshot, replace the directory wholesale from a fresh upstream checkout and record the new commit/date in this table within the same PR.

## References

- Liu, X. et al. (2012). *Toward a minimal representation of aerosols in climate models: description and evaluation in the Community Atmosphere Model CAM5.* Geosci. Model Dev., 5, 709–739. https://doi.org/10.5194/gmd-5-709-2012
- Liu, X. et al. (2016). *Description and evaluation of a new four-mode version of the Modal Aerosol Module (MAM4) within version 5.3 of the Community Atmosphere Model.* Geosci. Model Dev., 9, 505–522. https://doi.org/10.5194/gmd-9-505-2016

## License

This project is released under the GPL v3 License - see the [LICENSE](./LICENSE) file for details.
> [!IMPORTANT]
> Note that the vendored Fortran reference under `mam4-original-src-code/` retains its [upstream license](mam4-original-src-code/LICENSE).
