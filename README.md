# Toxic-Comment-Classification

This is an ML project which aims to classify a comment into either Toxic or Non Toxic class.  

I used Multinomial Event Model Naive Bayes for text classification. I learned this model during my online ML Course @ codingblocks.com  

- TCC_Model.ipynb file contains the code of the model.
- model.pkl is the pickle file created using pickle module to save the model.  
- app.py file is an API file which contains 1 endpoint '/predict', which accepts a comment and returns the class of the comment.