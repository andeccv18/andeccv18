# Autoregressive Novelty Detectors
### Anonymous ECCV18 submission - paper ID 2460

This repository serves as a demo of the behavior of Autoregressive
Novelty Detection, on the Shanghai Tech dataset.

The code has been developed in Pytorch 0.3.1 using a
Nvidia Titan X.

#### How to run the code.
To run the code, first download the pretrained weights
[here](https://drive.google.com/open?id=1arcMZguOg22j_d6wws2vDcV75YHVV0ow).

Than, you should call something like
````
python main.py --checkpoint_path=<path-to-checkpoint> --video_path=<path-to-video>
````
