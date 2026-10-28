# File: explain_main.py
# Usage: python explain_main.py  ===> example
#        python explain_main.py 42  ====> explain test-set order 42
# Description: entry point for Prolog explanations of an order predicted time

import sys
from src.prolog_explain import explain_prediction, explain_row

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Option A: an order number is given as parameter 
        explain_row(int(sys.argv[1]))
    else:
        # Option B: an example of order if no parameter is given 
        explain_prediction(ml_eta=32.4,
            distance_km=8.2,
            rating=4.1,
            weather="Stormy",
            traffic="Jam",
            festival="No",
            multiple_deliveries=2,)