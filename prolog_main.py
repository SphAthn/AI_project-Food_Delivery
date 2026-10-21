# File: prolog_main.py
# Usage: python prolog_main.py
# Description: Standalone execution pipeline for the Prolog Symbolic Engine.

from src.prolog_bridge import run_prolog_evaluation

if __name__ == "__main__":
    print("      STARTING PROLOG SYMBOLIC RULE ENGINE PIPELINE     ")
    
    run_prolog_evaluation()
    
    print("            PROLOG ENGINE PIPELINE COMPLETE             ")
