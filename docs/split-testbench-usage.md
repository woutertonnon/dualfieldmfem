# Split Testbench Usage (Initial Scaffold)

This repository now contains a first scaffold for the split architecture:

- `simbench_core`
  - `validate_case_v1(case)`
  - `apply_defaults_v1(case)`
  - `to_legacy_flat(case)`
  - `write_legacy_config(case, path)`
- `simbench_adapters.mfem_ns`
  - `MfemNavierStokesAdapter`
  - `MfemStokesAdapter`

## Minimal Example (Navier--Stokes)

```python
from simbench_adapters.mfem_ns import MfemNavierStokesAdapter

adapter = MfemNavierStokesAdapter("./build/hcurl_singlefieldnavierstokes_nitsche")

case = adapter.build_case(
    case_id="ConstantField_conv_order1_ref0",
    mesh="./geo/mesh/ConstantField.msh",
    outputfile="ConstantField/ConstantField_conv_order1_ref0",
    viscosity=0.001,
    functions={
        "force_data": "out[0] = 0;out[1] = 0;out[2] = 0;",
        "initial_data_u": "out[0] = 0;out[1] = 0;out[2] = 0;",
        "initial_data_w": "out[0] = 0;out[1] = 0;out[2] = 0;",
        "boundary_data_u": "out[0] = 1;out[1] = 0;out[2] = 0;",
        "exact_data_u": "out[0] = 1;out[1] = 0;out[2] = 0;",
        "exact_data_w": "out[0] = 0;out[1] = 0;out[2] = 0;",
    },
    order=1,
    refinements=0,
    dt=0.01,
    T=100.0,
    tol=1e-7,
    visualisation=1,
    printlevel=2,
)

config_path = adapter.write_config(case, "./data/config/ConstantField/ConstantField_conv_order1_ref0.json")
cmd = adapter.command_for(config_path)
print(cmd)
```

## Minimal Example (Stokes)

```python
from simbench_adapters.mfem_ns import MfemStokesAdapter

adapter = MfemStokesAdapter("./build/stokes_nitsche")

case = adapter.build_case(
    case_id="StokesTest_conv_order2_ref3",
    mesh="./extern/mfem/data/ref-cube.mesh",
    outputfile="StokesTest/StokesTest_conv_order2_ref3",
    mass=1.0,
    viscosity=0.02,
    functions={
        "force_data": "out[0] = 0;out[1] = 0;out[2] = 0;",
        "exact_data_u": "out[0] = x[0];out[1] = x[1];out[2] = x[2];",
    },
    order=2,
    refinements=3,
)

config_path = adapter.write_config(case, "./data/config/StokesTest/StokesTest_conv_order2_ref3.json")
cmd = adapter.command_for(config_path)
print(cmd)
```

## Notes

- The adapter currently writes the legacy flat JSON expected by existing C++ executables.
- This keeps current binaries unchanged while we migrate to canonical schema workflows.
