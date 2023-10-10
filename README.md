# MolSetRep

MolSetRep is a Python library that provides encoders and machine learning models for molecular set representation learning. The following models that are ready to be used with [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) are included:

- `LightningSRClassifier`
  - Wraps `SRClassifier`
  - Takes molecules encoded by `SingleSetEncoder` as an input
- `LightningSRRegressor`
  - Wraps `SRRegressor`
  - Takes molecules encoded by `SingleSetEncoder` as an input
