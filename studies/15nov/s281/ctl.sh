#!/bin/bash
# s281 control: plain fb, within0 s0 (task 0 of FAMS=fb) must reproduce runs/ttt/probs18/ttt18within0_fb_base-all_loso-s0.npz bit for bit
FAMS=fb exec bash studies/15nov/s281/cell.sh $((48 + $1))
