# Behavioral-Cloning-P3-USDC

In this project we train our computer to drive like us using simulator provided by Udacity [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip) which helps us generate images through driving it! These images are then feeded into model and the simulator is used to simulate our model. 

## Requirements 

* [Keras](https://keras.io) 
* [OpenCV](https://opencv.org)
* Numpy 
* Pandas 
* Sklearn 

## Model Training 

I first generated images using the training mode inside the simulator and drove the car using PS3 Joystick. It was difficult to drive using Keyboard since the sensitivity of angles are important for the project and for this reason , PS3 Controller using Better DS3 was used for training of Data. Model was trained only for first track (Not the jungle track). 

![The Images obtained by training]()

## Techniques and Augmentation 

It was important to realize what i required for the model to train. Scenery doesn't affect the way the car drives in long shot but the instantaneous decision based on the road. To draw an analogy we tell our car that don't try to look beyond the horizon.We also normalize like the way we've done in all our previous projects!  
So the techniques used here are , 
1. **Cropping** : The images which we recieve through training are basically a full sized image which includes scenery etc , So as mentioned before we are telling our car "don't look beyond the horizon" so we crop our image from the top and sides to focus on the roads and not get influenced by others.
![After cropping the Image]()

2. **Resize image** :- Depending upon our model we resize image such that we have easier training with lesser time training the model. 
  a. For VGG Model - 224 , 49 
  b. For NVIDIA model - 200 , 66
3. **Brightness** :- We want our model to make the most of the data which are available but we need to keep in mind that we can't feed the same images and train it since that would mean it could overfit , Secondly we need to realize that our simulator has constant lightening and hence , Inspired by this [post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9).
![After Brightness]()
4. **Flipping** :- We must realize that in our track we have a lot of left turn in comparision to the right turns. So we use Horizontal flipping , Horizontal flipping enables us to have more data and few unique data!. The basic idea is to create a mirror image and reverse the angles. "angle = angle*-1. 
![After Flipping]()

## Model 

In this case , I have tried two architectures and both code along with their output's are present in the repository. The first model is going to be a modified [End to End Learning for Self Driving Car](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and the second model is going to be a modified [VGG](https://arxiv.org/abs/1409.1556). 
I tried both the models and VGG outperforms NVIDIA model. But this could be due to the fact that NVIDIA model needs more training as it fails only at one particular curve from which it recovers and drives fairly right. On the other hand VGG model works perfectly for the track1 and fails at the track 2. 

### Model Architecture NVIDIA 

There are few changes made on the model's end as opposed to the original NVIDIA end to end architecture. 

The summary of the architecture i used is this :-
====================================================================================================
____________________________________________________________________________________________________
Layer (type)             |        Output Shape      |    Param   |   Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 31, 98, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       elu_1[0][0]                      
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 14, 47, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       elu_2[0][0]                      
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 5, 22, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       elu_3[0][0]                      
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 3, 20, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       elu_4[0][0]                      
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1, 18, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 1, 18, 64)     0           dropout_1[0][0]                  
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           elu_5[0][0]                      
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      flatten_1[0][0]                  
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        elu_6[0][0]                      
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         elu_7[0][0]                      
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          elu_8[0][0]                      
====================================================================================================
Total params: 252,219
Trainable params: 252,208
Non-trainable params: 11

As opposed to the original architecture which follows 
![NVIDIA End to End for Self Driving Car](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

Here , I've used the Lambda layer to Normalize the image and feed through the network.
The model was trained for 3 epochs with batch size being 128. Adam optimizer was used for the model with Mean Squared Error to check for loss. 

### Model Architecture VGG16 based. 

The idea was to build a model which could run in the lowest possible epoch due to machine constraint. I used VGG16 Architecture at first but it was really slow and the problem is more of what degree to steer instead of what direction to steer , So few changes were to be made. For this reason , I used 4 Layer convolutional network with relu along with 3 Fully connected layer's as my final output.I used normalized image and trained the data again on 3 epochs with batch size being 128 , loss function of mean squared error. 
Here's the architecture i used , 
====================================================================================================
Layer (type)      |               Output Shape  |        Param |    Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 49, 224, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 49, 224, 3)    12          lambda_1[0][0]                   
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 24, 112, 3)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 24, 112, 32)   2432        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 12, 56, 32)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 12, 56, 64)    51264       maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 6, 28, 64)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 28, 64)     36928       maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 3, 14, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2688)          0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 2688)          0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           1376768     dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 512)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 10)            5130        dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 10)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             11          dropout_3[0][0]                  
====================================================================================================
Total params: 1,472,545
Trainable params: 1,472,545
Non-trainable params: 0
____________________________________________________________________________________________________

The original VGG16 Architecture 

![VGG16](https://www.pyimagesearch.com/wp-content/uploads/2017/03/imagenet_vgg16.png)

I first had the idea to use Dropout after every layer but it was making the training slow and was pretty unnecessary as the Dropout at the end of Convolutions did the trick. 

### Training 

I trained the model on my computer with NVIDA 940MX 4GB GPU along with I5 7200U processor. I trained the model for 3 epochs which perfectly did the trick for the first track. With 10% Validation set. 

## Running the Model 

The model took around 15 minutes to train. There were changes which were to be made inside the drive.py file so that it doesn't throw index out of bounds error. 

### To run the model in NVIDIA Architecture - 
  ``` python modelnv.py ``` 
  
  Which would result in generation of h5 file which we are going to use for running. 
  
  ``` python drive1.py modelnv.h5 ```
### To run the model in VGG inspired Architecture - 

  ``` python model.py ```
  ``` python drive.py model.h5 ```
## Video of the car driving 
Using VGG at Speed 17 
![Running of the car at 17kmpl](https://youtu.be/14PhbDHrwVk) 
The model performed better at Slower speed using VGG best being achieved at speed limit set as 12 in the drive.py file. 

Unfortunately NVIDIA model doesn't make the cut as it leaves the circuit and then returns back at one particular curve (U-Turn). But i happen to infer that maybe a more epoch could do the trick , Which i leave it for future. 

## How to improve the output ? 
As stated earlier , I have trained the model using 3 epochs which in it's own way is less and would require atleast 5-6 to be a completely accurate model. On the other hand , It needs more data apart from the desert data. I feel that augmentation done allows the model to train precisely on the basis of the known curves that being said , Newer set of data could always improve the model in this case. 

## Future Extension 

The simulator doesn't actually have any other car's or obstruction apart from turns and steering a car in real life also depends upon the car's going at the front. Hence maybe trying to generate data using car games and running it sounds like a good option. Also , Throttle and Speed are predefined hence , We actually don't do anything to it , Maybe playing around with it could help example - In case of NVIDA architecture there were a lot of late turns leading to drifting of car.   
