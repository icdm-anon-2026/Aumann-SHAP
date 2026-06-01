\# MNIST experiment



\## Reproduction order

(Training is slow — skip if `resnet18\_mnist\_1vs7.pt` already exists.)



1\) `python ../run\_mnist.py --task train`

2\) `python ../run\_mnist.py --task equal\_split`

3\) `python ../run\_mnist.py --task micro\_game`

4\) `python ../run\_mnist.py --task heatmaps`

5\) `python ../run\_mnist.py --task patchtest`

6\) `python ../run\_mnist.py --task global`

7\) `python ../run\_mnist.py --task globalheat`



See `docs/REPRODUCIBILITY.md` for dependency details.

