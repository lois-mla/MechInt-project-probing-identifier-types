#  Probing and steering identifier types

This is a repository of code used in the experiments searching for the linear representation of identifier types in the embedding space of CodeLLMs. Further, it contains code used to prove causality of the given representation using steering. The results produced using this code are presented in the paper 'Are Identifier Types Identified? Linear probing of CodeLLMs for class, variable and function features'. The pdf of this report is in the mechint\_project.pdf file. 

The functions necessary to perform a linear probe on identifier types are in the linearprobe\_new.py file. Then, steering.py file contains code necessary to perform steering using the direction found with the probe. Both of those files are executed using train\_probe.py. The accuracy of the probe and steering results can be plotted using plotting.py. 

For the data we trained the linear probe on, which can be found in training\_data (a realistic-name contrastive dataset (data), a letter-name contrastive dataset (data\_final) and a letter-name non-contrastive dataset (data\_nocont)), the probes and results can be found in probes\_stored and figures. 
