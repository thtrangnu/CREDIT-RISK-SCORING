"""SAFE entry point for building the feature artifact. See the note at the end of
ml/src/features/build.py. This file ONLY imports main() and defines no classes, so
build.py is always loaded as a normal module rather than as __main__.
"""
from ml.src.features.build import main

if __name__ == "__main__":
    main()
