#!/bin/bash
python2 par_swarm.py -m vbsv2 &
python2 par_swarm.py -m arv1 &
sleep 21600
python2 par_swarm.py -m arv2
