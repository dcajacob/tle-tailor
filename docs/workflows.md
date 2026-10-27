# Scripts and notebooks

The example folder is now:

```text
tle_tailor/examples/
├── scripts/                 Python implementations and shared helpers
├── notebooks/               notebooks, including script-backed companions
│   └── maintained/          named maintained workflow companions
└── workflows.json           shared public workflow inventory
```

Start with the [deployment example](deployment.md) or [supplied-ephemeris fit](ephemerides.md). The simple deployment notebook is [`notebooks/deployment.ipynb`](../tle_tailor/examples/notebooks/deployment.ipynb); it calls the same implementation as `tle-tailor example`.

```bash
uv run --group notebooks jupyter lab tle_tailor/examples/notebooks
```

Select the project Python kernel. Script-backed companions call their named module, with an editable `arguments` list. Most default to `--no-plots`, which saves figures without opening windows; remove it to display them. Generated outputs go to `TLE_TAILOR_OUTPUT_DIR` or `./output`.

Scripts can also run as modules from the installed project environment:

```bash
uv run python -m tle_tailor.examples.scripts.gmat_integration.gmatpy_state_to_tle
```

Use `python -m tle_tailor.examples.scripts...`, rather than running a nested file by filesystem path. Installed module invocation works outside the source folder too. The old `tle_tailor.notebooks.*` module locations are replaced by `tle_tailor.examples.scripts.*`.

## Named workflow companions

The shared manifest inventories 14 older named workflows in addition to the compact deployment/ephemeris commands. Basic EQN and modern OPM examples default to gmatpy reference propagation. COE and basic no-BSTAR examples are SGP4 recovery/validation studies. In all cases the fitted TLE uses SGP4.

| Companion | Script module | Requirement |
| --- | --- | --- |
| [TLEFit - COE - FD](../tle_tailor/examples/notebooks/maintained/basic_examples/tlefit_coe_fd.ipynb) | `tle_tailor.examples.scripts.basic_examples.tlefit_coe_fd_script` | Maintained: inline TLE |
| [TLEFit - COE - JAX](../tle_tailor/examples/notebooks/maintained/basic_examples/tlefit_coe_jax.ipynb) | `tle_tailor.examples.scripts.basic_examples.tlefit_coe_jax_script` | Maintained: inline TLE |
| [TLEFit - EQN - FD](../tle_tailor/examples/notebooks/maintained/basic_examples/tlefit_eqn_fd.ipynb) | `tle_tailor.examples.scripts.basic_examples.tlefit_eqn_fd_script` | Maintained: inline TLE |
| [TLEFit - EQN - JAX](../tle_tailor/examples/notebooks/maintained/basic_examples/tlefit_eqn_jax.ipynb) | `tle_tailor.examples.scripts.basic_examples.tlefit_eqn_jax_script` | Maintained: inline TLE |
| [TLEFit - EQN - No BStar](../tle_tailor/examples/notebooks/maintained/basic_examples/tlefit_eqn_no_bstar.ipynb) | `tle_tailor.examples.scripts.basic_examples.tlefit_eqn_no_bstar_script` | Maintained: inline TLE |
| [GMAT OPM Prop FD Example](../tle_tailor/examples/notebooks/maintained/gmat_integration/gmat_opm_prop_fd_example.ipynb) | `tle_tailor.examples.scripts.gmat_integration.gmat_opm_prop_fd_example_script` | Maintained: inline state; gmatpy default |
| [GMAT OPM Prop JAX Example](../tle_tailor/examples/notebooks/maintained/gmat_integration/gmat_opm_prop_jax_example.ipynb) | `tle_tailor.examples.scripts.gmat_integration.gmat_opm_prop_jax_example_script` | Maintained: inline state; gmatpy default |
| [TLE from GMAT](../tle_tailor/examples/notebooks/maintained/gmat_integration/tle_from_gmat.ipynb) | `tle_tailor.examples.scripts.gmat_integration.tle_from_gmat_script` | Historical: separate legacy API |
| [TLEFit - EQN - GPS FD - GPS Sat](../tle_tailor/examples/notebooks/maintained/gps_examples/tlefit_eqn_gps_fd.ipynb) | `tle_tailor.examples.scripts.gps_examples.tlefit_eqn_gps_fd_script` | Maintained: source-only ephemeris |
| [TLEFit - EQN - GPS JAX - GPS Sat match - import](../tle_tailor/examples/notebooks/maintained/gps_examples/tlefit_eqn_gps_jax_match.ipynb) | `tle_tailor.examples.scripts.gps_examples.tlefit_eqn_gps_jax_match_script` | Maintained: source-only ephemeris |
| [TLEFit - EQN - GPS JAX - GPS Sat](../tle_tailor/examples/notebooks/maintained/gps_examples/tlefit_eqn_gps_jax.ipynb) | `tle_tailor.examples.scripts.gps_examples.tlefit_eqn_gps_jax_script` | Maintained: source-only ephemeris |
| [TLEFit - EQN - GPS - GPS Sat - No BSTAR](../tle_tailor/examples/notebooks/maintained/gps_examples/tlefit_eqn_gps_no_bstar.ipynb) | `tle_tailor.examples.scripts.gps_examples.tlefit_eqn_gps_no_bstar_script` | Source ephemeris; ~130 m fit accepted, long-hold-out limits apply |
| [TLEFit - EQN - GPS FD - Icesat](../tle_tailor/examples/notebooks/maintained/icesat_examples/tlefit_eqn_gps_fd_icesat.ipynb) | `tle_tailor.examples.scripts.icesat_examples.tlefit_eqn_gps_fd_icesat_script` | Maintained: source-only ephemeris |
| [TLEFit - EQN - GPS JAX - Icesat](../tle_tailor/examples/notebooks/maintained/icesat_examples/tlefit_eqn_gps_jax_icesat.ipynb) | `tle_tailor.examples.scripts.icesat_examples.tlefit_eqn_gps_jax_icesat_script` | Maintained: source-only ephemeris |


## Inputs and historical studies

GPS/ICESat scripts need caller-owned ephemerides, excluded from packages. Existing studies use `TLE_TAILOR_DATA_DIR` with an `ephemeris/<filename>` layout; the new generic [ephemeris command](ephemerides.md) takes an explicit file. The retained GPS fixed-BSTAR fit is about 130 m RMS; its long-hold-out limit is separate, as explained in [accuracy](accuracy.md).

Original research notebooks are retained separately from the maintained companions. Their extra studies may require source-only data, catalogs, research dependencies or an external GMAT API. Installing notebook support does not make every historical study runnable. Local mission data and proprietary workflows are not public examples.

Shared input assets use the existing `tle_tailor/data` directory or an external `TLE_TAILOR_DATA_DIR`; BSP copies are consolidated at `data/spice/de421.bsp`. Three distinct historical GMAT assets live under `data/examples/gmat_integration`, without overwriting same-named ephemerides. Inputs are not generated-output locations. Historical notebook recovery checkpoints remain local and unchanged.
