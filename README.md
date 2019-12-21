# cnn-implementations

I attempted some Kaggle competitions for CNN (Convolutional Neural Networks) and the scrips are uploaded here. Most of the results here ended up middle of the pack in the Kaggle results hierarchy but the point was more to learn end-to-end implementations including data manipulation, model selections and results views with tensorflow rather than try to win 


I try to code with increasing complexity with each attempt


CNN_on_the_cloud - this allowed me to initalize and run a complete model with training on the cloud - I used google compute engine because they gave me credits for free 


mnist digits classifer - this is often considered as the "hello world" of deep learning. I used a basic CNN here. pretty cool accuracy within just 30 lines of code 


cats_dogs_CNN - I attempted to learn transfer learning here by taking the first 10 layers from VGG16 trained on imagenet and then training a fully connected last layers. 
My model basically overfits (train accuracy ~99% and validation accuracy ~ 85%) but good intro to transfer learning 



