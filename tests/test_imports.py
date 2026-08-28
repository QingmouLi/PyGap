
def test_imports():
    from pathlib import Path
    import sys
    project_root = Path(__file__).resolve().parents[1] # Path.cwd().parent
    sys.path.insert(0, str(project_root))

    import numpy as np
    import pandas as pd
    import seaborn as sns    
    from scipy.stats import skew, kurtosis,gaussian_kde, probplot

    from PyGap import dataSamples as ds
    from PyGap import EigenSig 
    from PyGap import dataSamples
    from PyGap import LSHE
    from PyGap import utility as util  
    assert EigenSig is not None, 'Can not create EigenSig Object' 

if __name__ == "__main__":
    test_imports()
    print("PyGap had been successfully tested")
