# DFC model source

`DFCModel.cu` supports the settling and particle-flow stages in
`demo/newton_dem/robotic_3d_printing/demo_dfc_3d_printing.py`.

It was retrieved from
[`Ruochun/pyDEME_tests`](https://github.com/Ruochun/pyDEME_tests/blob/main/DFCModel.cu),
where `pyDEME_DFCSlump.py` demonstrates its DEME configuration and settling
workflow. The copy is kept with the demo because it is JIT-compiled by DEME at
runtime and defines this scenario's physical contact law.

The parameter values and mortar-coated particle construction used by
`demo_dfc_3d_printing.py` follow the CPU `Demo_DFC_MiniSlump` at Chrono commit
`406c9efeac07ccd4123993d03bb6aa0ad665e8c7`, rather than the calibration in
`pyDEME_DFCSlump.py`. The pyDEME source supplies the compatible GPU kernel;
the MiniSlump source supplies the scenario calibration.
