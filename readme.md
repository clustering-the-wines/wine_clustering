# Wine Quality Clustering and Predictions
 
# Project Description

A small region in northern Portugal is known for its production of various types of wine. Wines produced in this region go through an extensive testing process where different chemical attributes are measured and recorded. These measurements were recorded and placed in a dataset. This dataset was analyzed to come up with the features that would be the best predictors of wine quality. Wine quality was measured on a scale of 3-9 with 3 being low quality and 9 being high quality. After most important features were identified, features were applied to a machine learning algorithm to predict wine quality.

# Project Goal
 
* Discover features that have the strongest influence on wine quality.
* Use features to develop a machine learning model to predict wine quality.
* This information can be used to identify features that can be focused on in the wine making process to ensure highest wine quality.
 
# Initial Thoughts
 
My initial hypothesis is that the sugar level influences wine quality the most and that red wine, in general, has a higher quality.
 
# The Plan
 
* Aquire data from data.world
 
* Prepare data
   * Create Engineered columns from existing data
       * fixed_acidity
       * volatile_acidity
       * citric_acid
       * residual_sugar
       * chlorides
       * free_sulphur_dioxide
       * total_sulphur_dioxide
       * density
       * pH
       * sulphates
       * alcohol
       * quality
       * wine_type
 
* Explore data in search of drivers of tax value
   * Answer the following initial questions
      * How many of wines of each quality are there?
      * Is there a relationship between fixed and volatile acidity?
      * Is there a relationship between residual sugar and wine quality?
      * Does pH and alcohol content have a relationship?
      * Does sulphates influence wine quality?
      * Does wine type influence wine quality?
      * Is there a relationshp between sulphates and sulphur dioxide?
      
* Develop a clustering Model based on fixed acidity and volatile acidity
      * developed clustering model based on free sulphur dioxide and total sulphur dioxide
  
* Use features identified in explore build predictive models of different types
   * Evaluate models on train and validate data
   * Select the best model based on accuracy
   * Evaluate the best model on test data

* Use features and clusters to see if predictive models improve in accuracy

* Draw conclusions
 
# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|fixed_acidity| most acids involved with wine or fixed or nonvolatile|
|volatile_acidity| the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste|
|citric_acid| found in small quantities, citric acid can add 'freshness' and flavor to wines|
|residual_sugar| the amount of sugar remaining after fermentation stops|
|chlorides| the amount of salt in the wine|
|free_sulphur_dioxide|the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine|
|total_sulphur_dioxide| amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine|
|density|the density of water is close to that of water depending on the percent alcohol and sugar content|
|pH| describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale|
|sulphates|a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant|
|alcohol|the alcohol content of the wine|
|quality|the quality of wine based on a scale of 3-9|
|wine_type|whether the wine is white or red|


# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from sql database
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
# Takeaways and Conclusions

 
# Recommendations

