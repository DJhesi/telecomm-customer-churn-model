# SyriaTel: Know Your Customer

### David Johnson and Elina Rankova

  <p align="center"><img src="https://media.licdn.com/dms/image/C4D12AQH-Qk_eTZv6iA/article-cover_image-shrink_720_1280/0/1622638706496?e=1715212800&v=beta&t=a3k-gWfqrlzfv7inhVBpUU9xxPjnJ0of4viF4tFu-Oc" width="720" height="450" style="margin: 0 auto;"/></p>

<u>image source</u>: <a href="https://www.linkedin.com/pulse/churn-analysis-smriti-saini/">Churn Analysis Article</a>

## Business Problem and Understanding

**Stakeholders:** Director of Member Operations, Member Operations Manager, Member Retention Manager, Member Support Manager

The business problem at hand is to predict customer churn for SyriaTel, a telecommunications company, in order to minimize revenue loss and enhance customer retention efforts. With customer attrition posing a significant challenge to profitability in the telecom industry, SyriaTel seeks to identify patterns and trends within its customer base that indicate potential churn. By leveraging historical data and predictive modeling techniques, the aim is to develop a classifier that can accurately forecast which customers are likely to discontinue their services, enabling SyriaTel to implement targeted retention strategies and ultimately strengthen its competitive position in the market.

**The goal:** Create a model to predict churn in telecom members contacting support. We are aiming to reduce the amount of cases in which members are mistakenly identified as retained (false negative) vs mistakenly identified as churned to ensure we capture all members who may churn (positive).

## Data Understanding and Exploration

For this analysis, the SyriaTel churn data was sourced from <a href = "https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset">Kaggle</a>

The dataset contains data on the customers of a Telecom company. Each row represents a customer and the columns contain customer’s attributes such as minutes, number of calls, and charge for each time of day and international. In addition we also have information about the customer's voicemail and customer call behavior.

### Observations:
- The dataset has no missingness and most columns are numeric
- Of the 3,333 customers in this dataset, 483 terminated their contract with SyriaTel
- Transforming target `churn` as well as `international_plan` and `voice_mail_plan` to binary 0/1 is needed
- `state` appears numeric but is actually categorical and needs to be transformed as well
- `phone_number` can be dropped as there are no duplicate entries

#### Class Imbalance
This is an imbalanced dataset, with 14.5% of customers lost, balancing will be necessary. 

<p align="center"><img src="Images/Class Imbalance.jpeg" /></p>

#### There are several correlations worth noting:
`total_intl_charge`, `total_day_charge`, total_eve_charge`, and `total_night_charge` is perfectly correlated with `total_intl_minutes`, `total_day_minutes`, total_eve_minutes`, and `total_night_minutes` respectively. This makes sense since the company is charging by the minute. 
> If we need to, we can confidently drop the 'charge' column from each category; day, eve, night, and intl. We can keep the 'minutes' category as it is unclear what currency metric 'charge' is referring to.

In addition, there is a near perfect correlation between `number_vmail_messages` and `voice_mail_plan`, this makes sense and these two columns much like 'charge' and 'minutes' are telling us the same thing. 
> If we need to, we can drop `number_vmail_messages`.

Lastly, there are a couple of weak correlations associated with our target `churn` variable; It seems `customer_service_calls`, `international_plan` and `total_day_minutes` have a slight positive correlation with churn. 
> While weak correlations, we would want to consider including these features in our models.

<p align="center"><img src="Images/Data Exploration Heatmap.jpeg" /></p>

## Data Preperation

To prepare the data for modeling, several steps had to be taken as described below. All train/test splits maintain the default .75/.25 proportion respectively. We know we have class imbalance, so we will have `stratify = y` so our class proportions stay the same for both our train and test data.

### Model 1 & 2
Given our selected approach to these `LogisticRegression` models, we had slightly different steps applied depending on the model. All train/test splits maintain 

#### Pre-Split

Before splitting our data between train and test, we performed some simple processing:

- Since the column names were formatted with a space between words, we transformed them to include and underscore as per column name standard formatting
- `LabelEncoder` was used to perform transformtions on the following categorical columns:
  - `churn` orriginally in binary True/False format
  - `international_plan` and `voice_mail_plan` in binary Yes/No format
- Dropped `phone_number` column as there were no duplicate entries as mentioned previously

#### Post-Split

After splitting our data into a train and test we had to perform a couple of other transformations depending on the model criteria.

For the first two models we used `OneHotEncoder` to transform the `area_code` and `state` categorical columns to numerical format. This left us with an `X_train` containing 69 features. In addition, we used `SMOTE` to resample our data and handle class imbalance.

**Model 2 with `SelectFromModel`** to aid with important feature selection we called on this meta-transformer to reduce our 

### Model 3

For our 3rd model we took a manual approach and redefined the DataFrame criteria which lead us to having to conduct a fresh train/test split.

#### Pre-Split

We decided to only include highly correlated variables since we had previously stated there were some features which had extremely high correlations with eachother. For this model we were left with only the features seen below in a heatmap no longer demonstrating any co-linearity.

<p align="center"><img src="Images/Model 3 Heatmap.jpeg"/></p>

Since tranforming column names applies generally to the dataframe, we did not have to repeat this as it was already complete as the first pre-processing step.

#### Post-Split

Since we redefined a new `X` and `y` we also applied `SMOTE` to this fresh data set, creating reduced versious of our training data. 
> Sinced we eliminated any categorical columns in need of transformation, the `OneHotEncoder` was not necessary for this model.

## Modeling

Focusing on predicting churn, we will focus on finetuning to the `recall` metric to ensure we are predicting as many True Positive results (customers predicted to churn who churn) and reducing False Negative (customers predicted to be retained who churn) as much as possible. 

Our business initiatives are not high risk so a somewhat disproportionate amount of False Positives (customers predicted to churn who are retained) is tolerated and approach to this sect of customers will be addressed within our evaluation and recommendations.

### Base Model
We build our base model with `DummyClassifier` using the `stratified` strategy since we have an imbalanced dataset skewed in the direction of class 0 when we are interested in predicting class 1. We do not apply `SMOTE` here to get truly baseline results.

From the get go our Base model produced an average `accuracy` score of ~0.75. This is a good start and gives us confidence to proceed with improving our `recall` and still maintianing fairly balanced results.

### 1st Model
To start, we evaluate a basic `LogisticRegression` model, before applying `SMOTE`. We use the detault L2 penalty with this initial model.

**Comparing our first `LogisticRegression` model with our `base`**, we can see that our `LogisticRegression` model does somewhat better at predicting `churn` with a higher True Positive Rate than our `base`.

<p align="center"><img src="Images/Base vs Log.jpeg"></p>

To choose the right solver, we run this model with both L1 and L2 solvers. It looks like the Logisic L1 model does better than both previous models but only slightly. However our class imbalance makes it difficult to assess accurately and needs to be addressed.

After applying `SMOTE` with an even 0:1 split, we cross validate our model with the `ModCrossVal` class created to make cross validation an easier process, in which we specify `scoring = 'recall'`. Our model performs nearly the same on the train and test (validation) data. We can probably get this even higher after we simplify our model some more.

<p align="center"><img src="Images/1st Model CV.png"></p>

**Finetuning `C` with Cross Validation:** Creating a loop to test out `C` values `[0.0001, 0.001, 0.01, 0.1, 1]` we find that the lowest `C` yields the highest `recall`. 

Our optimized results after finetuning the `C` look pretty good, though around the same as before optimization. Once we attempt to simplify some more, we will want to look at other scores such as accuracy and precision to make sure our results are balanced enough for the business problem at hand.

<p align="center"><img src="Images/1st Model CV Optimized.png"></p>

### 2nd Model
As prevously stated, we know that there are features that are highly correlated. We use `SelectFromModel` to select features for us that are most important. After additional preprocessing with `SelectFromModel` we run and cross validate using the same `ModCrossVal` class.

We will use the default threshold to start and identify which features meet threshold requirements. Since we are still using our L1 Logistic model, the default threshold will be $1e^-5$. It looks like there are several features that do not meet the threshold.

**Before finetuning** our selected feature model did around the same as our Logistic L1 model before finetuning. It is worth noting that this is a simpler model as it has reduced features. 

**Finetuning `C` with Cross Validation:** Just like our Logreg L1 model and using the same test `C` values, the Logreg Select model does best with smaller `C` values, so we will want to use the smallest value with our optimized model.

Our Logistic Select model did pretty well! It performed slightly better at recall than our first Logtistic L1 model.

<p align="center"><img src="Images/2nd Model CV Optimized.png"></p>

<p align="center"><img src="Images/Log L1 vs Log Select.jpeg"></p>

### 3rd Model

For our final itteration of the LogisticRegression model we should try manual feature selection with features we know to be highly correlated with `churn`. 

Here we redefined our DataFrame:

<p align="center"><img src="Images/3rd Model DataFrame.png"></p>

 **Before finetuning** and after performing a new split and re-applying `SMOTE` to the fresh data, we run our results. Our model performs slightly worse than our previous two.

**Finetuning `C` with Cross Validation:** Using a different set of tests `C` values `[0.00015, 0.0002, 0.0015, 0.002, .015]` , the smallest `C` values gives us the best results. We will again, use the smallest value within our optimized results.

We get an extremely high recall score after optimizing! We will definitely want to make sure we balance accuracy within our decision making process. All in all, it seems like our manual feature selection yields the best recall.

It is also great to see that our bias and variance are balanced as our train and validation performance on all models is mostly even.

<p align="center"><img src="Images/3rd Model CV Optimized.png"></p>

### Compare Optimized Logistic Models

Comparing confusion matrices of all 3 `LogisticRegression` models, our most recent Logistic Reduced model does best at predicting True Positives (customers going to churn) and reducing False Negatives (customers appearing to be retained but who actually churn).

This can provide valuable intervention insights to our stakeholders given a strategic approach to address the high amount False Positives (customers appearing to potentially churn but actually end up retained).

<p align="center"><img src="Images/Logistic Model Comparison.jpeg"></p>

### Run Final Models on Test

We now run our models with test data and evaluate each classification report associated. As expected, our 3rd Model produces the highest recall. As this is our primary focus for Phase 1 of this business initiative we will want to recommend deployment of this model and address the concerns regarding our lower precision and accuracy scores within our approach recommendations as well as next steps.

<p align="center"><img src="Images/Classification Report.png"></p>

## Final Evaluation & Conclusion

After careful consideration we are recommending to implement Model 3 with manual feature selection of highly correlated variables. This model provides the highest Recall or True Postive Rate and most closely satisfies the goals of <ins>_Phase 1_</ins> of this business initiative. Below we go into detail regarding this decision including additional recommendation on intervention approach.

### Recommendations:

As this is <ins>_Phase 1_</ins> of the project, we are hyper focused on identifying True Positive cases while reducing False Negative instances. Therefore, we are primarily focused on recall or true positive rate.

To account for our recall-focused path, a variety of low touch to high touch engagement models is recommended to account for the high number of False Positives within these models. An automated low touch model to start and gather data on customer satisfaction of those predicted to churn will yeild best results. Acting accordingly with a scaled  approach given the feedback collected will be crutial and create a positive customer experience for all.

**Positive Implications:**

<ins>_Customer Retention:_</ins> High recall means that your model is effective at identifying customers who are likely to churn. This allows the business to proactively intervene and take steps to retain these customers, such as offering incentives, personalized promotions, or improved customer service.

<ins>_Reduced Churn:_</ins> By effectively targeting at-risk customers, you may be able to reduce the overall churn rate, leading to increased customer retention and long-term profitability.

**Negative Implications:**

<ins>_Costs:_</ins> A low precision score means that there may be a significant number of false positives, leading to unnecessary costs associated with retaining customers who were not actually at risk of churning. These costs may include incentives or discounts offered to retain customers. 

<ins>_Customer Experience:_</ins> Misclassifying customers who were not actually at risk of churning as "churners" may lead to unnecessary interventions or communications, potentially impacting the customer experience negatively.

**Data Limitation and Future Considerations:**

When looking to optimize our results and produce the most accurate prediction of customers who are likely to churn, we find that it may be best to use a combination of classifier models to balance precision and recall. However, given the need to edit the training data, this posed an issue. 

In <ins>_Phase 2_</ins>, we would recommend gathering additional data to account for class imbalance and revising which feature hold importance in relation to churn. 

By simplifying the data before modeling, we are more likely to yield positive results and open up options to combine models using the same training data for a more balanced learning mechanism.
