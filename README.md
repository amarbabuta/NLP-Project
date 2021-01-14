
### Climate Change Misinformation Detection


### 1. What is the task? 
A system that detects whether a document contains climate change misinformation using NLP.




In this project, the task is to predict climate change misinformation from the document of test data from training data, which comes from an in-class [Codalab Competition](https://competitions.codalab.org/competitions/24205). Our works include prepairing the data, data preprocessing, feature engineering, model selection and ensemble models etc. For more details, please check the [project specifications](https://github.com/amarbabuta/NLP-Project/blob/master/project%20.pdf) and [project report](https://github.com/amarbabuta/NLP-Project/blob/master/COMP90042-%20Project%20report.pdf).

### 2. Data
The `Data` contains both train data and test data.
#### 2.1. Original Data
`train.json`
> _The original training dataset which contains 1168 text. The training data contains labels with value 1 meaning trainin gdocument provided are articles with climate change miosinformation. So external data was added of label 0 to make it a binary classifiction problem._



`test-unlabelled.json`
> _The original test dataset which contains 1410 text of label 1 as well as label 0._

`external.json`
> _The external dataset which contains 4168 text of label 0._


`dev.json`
> _The dev dataset which contains 100 text of label 01 and 0._







`all_clean_data.csv`
> _The entire processed training dataset which contains 328932 tweets posted by 9297 users._

`test_clean_data.csv`
> _The entire processed test dataset which contains 35437 tweets posted by the same user group in the training dataset._

`train.csv`
> _The random 9/10 processed training dataset used for partial training dataset._

`train.csv`
> _The random 1/10 processed training dataset used for partial test dataset._

### 3. Code
#### 3.1. Data Preprcessing and Feature Engineering
`project_!_NLP.ipynb`
#### for experimental purpose
##### 3.1.1 Normal Features
> _is used for data preprocessing including non-English characters (e.g. emoticons and punctuations) and stopwords, as well as word tokenization and lemmatization based on [nltk](https://www.nltk.org/) package for training, dev as well as test data._

Applying normal features data to train the model with Bernoulli Naive Bayes.



##### 3.1.2 YAKE(Yet Another Keyword Extraction) for Bag of words

< _is used for data processing with Bag of Words by apply based on [yake](https://github.com/LIAAD/yake) package for training, dev and test data._



Before entering data into the models，using **TF-IDF** to transfer text into a vector or matrix. This process is implemented by `TF-IDF Vectorizer` and `TfidfTransformer` modules from [_scikit-learn_](https://scikit-learn.org/stable/) package.


And then applying extracted features with the linear SVC model to training data.

#### Final Approach used
##### 3.1.3 Tf-IDF Vectorizer

Before entering data into the models，using **TF-IDF** to transfer text into a vector or matrix. This process is implemented by `TF-IDF Vectorizer` and `TfidfTransformer` modules from [_scikit-learn_](https://scikit-learn.org/stable/) package.



#### 3.2. Model Selection
Five machine learning/deep learning models based on [_scikit-learn_](https://scikit-learn.org/stable/) and [_Keras_](https://keras.io/) are implemented in this part, including the **_Multinomial Naive Bayes_**, **_SVC_**, **_Bernoulli Naive Bayes_**, **_Logistic Regression_** and **_Linear SVC_**.

* - Multinomial Naive Bayes Model.
* - Bernoulli Naive Bayes Model.
* - Logistic Regression Model.
* - Linear Support Vector Classifier Model.
* - Support Vector Classifier Model.

#### 3.3. Ensemble Learning

> _Ensemble learning is a powerful technique to increase accuracy on a most of machine learning tasks. In this project, we try a simple ensemble approach called weighted voting to avoid overfitting and improve performance. The basic thought of this method is quite simple. For each prediction from the results of different models, we give them a weight corresponding to their individual accuracy in the previous stage. If the predicted labels of two models are the same, we just add their weight together. Then we select the prediction with highest weight as the final prediction._

Considering the individual performance of the previous models, we try three different combinations: 
* SVC + LR + MultinomialNB.
* LR + MultinomialNB + BernoulliNB.
* linearSVC + MultinomialNB + LR.

#### 3.3. Hyperparameter Tuning

Hyper parameter Tuning of the best classifier which is for Logistic Regression was used based on grid search technique to predict the labels for test data.

### 4. Future Works
Due to the limitation of time, we have some ideas which might be worthy but have not yet to try:
* Using [SMOTE](https://arxiv.org/pdf/1106.1813.pdf) algorithm to deal with the unbalanced training dataset.
* Hyper-parameter optimization based on grid search technique.
* Adjusting the weight of penalty item. Giving large value when the prediction for minority class is wrong and small value when the  prediction for majority class is wrong.
* Some more complicated but powerful ensemble learning methods which can be found [here](https://mlwave.com/kaggle-ensembling-guide/).
