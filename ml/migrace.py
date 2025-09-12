import glob
import os

from joblib import dump, load

for path in glob.glob(os.path.join("*.pkl")):
    model = load(path)  # načti do RAM
    new_path = os.path.splitext(path)[0] + ".joblib"
    dump(model, new_path, compress=0)
    print(f"Converted {path} -> {new_path}")
