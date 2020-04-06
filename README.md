# mobileye-detect-traffic-light-gogoplodeske
mobileye-detect-traffic-light-gogoplodeske created by GitHub Classroom
This project dealing with detecting trafic lights on images ( frames of video taken from a camera from the car) and estimating their distance in 
the rel world from the car. 

It takes as input images and for each image it loactes where the trafic lights in the image and how far it is from the car as an example 
![GitHub Logo](/NN.png)
Format: ![Alt Text](url)
Links

We can see there is a trafic light detected on 24m distance. 

The pipline of the project is done by following: 

1) the run.py it runs the project. there you can determind from which dirictiry to take the frames and how many frames to process and more 
parameters. 
the run.py it runs then the cotroller whcih iterate over the frames and call TFL_Man class. The TFL_Man when recving a frame it update the 
previous frame to take its currentframe and the current take the one it recieves. We need to perserve the previous frame 
becuase we use in part3 to calculate the location by the projection and epipolar priciple. In addition TfL_Man runs Part1 which calculte the candidate using image processing tools. We need this candidate, in Part2, to crop batches to feed to the neural network which tells if 
a batch include a trafic light or no if yes then in Part3 calculate its distance from the car. Here is a display for this design: 

![GitHub Logo](/design.png)
Format: ![Alt Text](url)
Links


and here more details for Part1,Part2 and Part3:
![GitHub Logo](/details.png)
Format: ![Alt Text](url)
Links

Comments: 
In part2 you need a trained nerural network. I trained a model and saved in traing. and due to its huge sixe cauldn't be able to 
load it in github. 
