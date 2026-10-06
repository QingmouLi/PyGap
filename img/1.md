# Fill data gaps using Eigenspace & Fourier series

Python code is developed (PyGap) to fill data gaps (continously losed data), detect/seperate anomalies, data quality control. PyGap includes EigenSig and LSHE methods working in eigenspace and using Fourier series, respectively
- (1) Data gap filling
stripped areas are synthetic data gaps (geomagnttic Y component), filled gaps using EigenSig and LSHE are shown with dashed lines.
- (2) residuals of EigenSig and LSHE
- (3) Singular Spectrum Analysis (SSA) in EigenSig
- (4) Coordinates in Eigenspace 
Epoch traces in EigenSpace, red symbols are observed, black ones are traced using bilinear method.
- (5)basis features in eigenspace 
first basis in the eigenspace, commonly it owns a great number of total energy for physical signals. Here for Y, it owns >99% of total energy
- (6) Significant frequencies 
detected frequency components in LSHE, it detects significant frequency components with refined frequency resolution and struct statistical testing.
- (7) Residual historgram distribution statistics between EigenSig and LSHE
- (8) Residual violin charts between EigenSig and LSHE. 
