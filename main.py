from utils.io import *

def main():
    
    first_sequence = Sequence("training", 1)
    init_pos, init_vel, init_orientation = first_sequence.get_initial_state()
    
    for ts, acc, ang_vel, img, baro_height, gt in first_sequence:
        pass
    
if __name__ == "__main__":
    
    main()