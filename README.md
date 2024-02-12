# FGSM - Protection and Detection
Marius Stokkedal & Sebastian Bengtsson

This report is a further development of the report [2]. 
The work is intended to develop a defense for image classifier of aerial photos - in this case a Convolutional neural network (CNN) model. 
The attack model used is the Fast Gradient Sign Method (FGSM) and the models that are used for defense are: Support Vector Machine (SVM), k-Nearest Neighbors (KNN) and Decision Tree (DT). 
The first attempt at defense models does not work as hoped - most of the of the data is classified as being under attack regardless of whether it is or not. Splitting the data produces more promising results, but it is only being explored as a concept and to build a foundation for future research.
