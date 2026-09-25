import sys
import pandas as pd
import numpy as np
import sklearn


def main():
    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION")
    print("STAGE 0 - ENVIRONMENT CHECK")
    print("=" * 60)

    print("\nPython:")
    print(sys.version)

    print("\nPandas:")
    print(pd.__version__)

    print("\nNumPy:")
    print(np.__version__)

    print("\nScikit-learn:")
    print(sklearn.__version__)

    print("\nEnvironment check complete.")


if __name__ == "__main__":
    main()
