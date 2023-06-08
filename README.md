# CSE455


Participated in the Kaggle Bird Competition. The data is in the birds folder with all the train data being in train and the test data being in test. The test data doesn't have labels which is what we submit to Kaggle. names.txt and sample.csv are some helper files for identifiers. My code is in test.py while my presentation is in 2023-06-07 23-58-09 mkv. I do have some csv of predictions but they've generally fared relatively poorly i.e. ~50 classification accuracy. due to be stuck in local minima



Generaly worked well with simpler neural network architectures because teh more complex ones would get stuck at local minima even with constant learning rates. I think the scope of my project
was too high initially so I took a lot of time tuning a lot of hyperparameters for the complex neural networks instead of starting from a simple one. My accuracies were generally quite poor as a result. It was finals week and the training took quite long as I was adjusting the hyperparameters. I either shouldve started with a simpler neural network or tested my neural networks' efficacy on a smaller subset of data that I trained continuously on. I do think however, that the AdamW and CyclicLR will work better to get out of the local minimas that I got stuck in and couldn't get out of even with more epochs. 

I don't think my pretrained model or data augmentation techniques led to any problems in convergence of optimal parameters here given that even if I adjusted it, my neural networks would still get stuck in the local minima. 
