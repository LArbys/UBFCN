#Segmentation in LArTPC

see develop branch

###ub-fcn-alexnet
alexnet backbone fcn, gradient flow blocked by one of the layers.

###ub-fcn-small
alexnet backbone fcn but I chop off conv4 and conv5 and use xavier initialization instead of gaussian. Loss goes down and I see L1 and L2 norm changing as well. I see weights change at each step by some non-negligible value. After ~100 steps (~3200 examples) I start to see drop in background score around particle.

![alt text](http://www.nevis.columbia.edu/~vgenty/public/o_0.png "0")  
![alt text](http://www.nevis.columbia.edu/~vgenty/public/o_1.png "1")  
![alt text](http://www.nevis.columbia.edu/~vgenty/public/o_2.png "2")  
![alt text](http://www.nevis.columbia.edu/~vgenty/public/o_3.png "3")  
![alt text](http://www.nevis.columbia.edu/~vgenty/public/o_4.png "4")  
![alt text](http://www.nevis.columbia.edu/~vgenty/public/o_5.png "5")  