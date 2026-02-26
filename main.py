# main.py

from config import default_config
from algorithms.ea_connectivity import train_ea

def main():
    final_P = train_ea(default_config)
    # optional save here

if __name__ == "__main__":
    main()