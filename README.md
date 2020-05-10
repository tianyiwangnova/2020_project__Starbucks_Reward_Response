# Starbucks Campaigns Analysis

*Tianyi Wang*
*2020 May*

This is a Capstone project for Udacity Data Science Nanodegree parterning with Starbucks. In this project, we have simulated datasets that mimic customer behavior on the Starbucks rewards mobile app. We will combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type.

## Background

Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an informational advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks and not all users receive the same offer. Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. Someone using the app might make a purchase through the app without having received an offer or seen an offer.

To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.

However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.


## Data

Exploratory data analysis can be found in the notebook `01 Data Exploration`. There are 3 tables:
<br>`portfolio`: information of the offers 
<br>`profile`: demographic information of the customers
<br>`trancript`: transactions
<br>Let's take a quick look at each table:

### Portfolio

There are 10 different promotions that belong to 3 general types: informational, buy-one-get-one and discount. There are 4 features that describe the promotions: channels, difficulty (money required to be spent to receive reward), duration and reward. We found out that **discount promotions usually have long durations and high difficulties while buy-one-get-one promotions have shorter durations and lower difficulties. Discount promotions will have higher reward.** In general, longer promotions will have higher difficulties.

![promos](https://raw.githubusercontent.com/tianyiwangnova/2020_project__Starbucks_Reward_Response/master/screenshot/promos.png)

### profile

`Age`: 12.8% of the data are missing. About half the customers are between 50 and 80 years old.
<br>`Gender`: 1.4% of the data are missing. 57.2% of the customers are males.
<br>`Customer age (how long the customer has been with us)`: 56% of the customers joined in the past 3 years.
<br>`Income`: 12.8% of the data are missing. Income seems to be corrrelated with age. In this simulated dataset, different age buckets have different limits of maximum incomes. For people under 40 years old, their income won't exceed 
80k and for people under 50 years old, their income won't exceed 100k. For older people, the maximum income is about 120k. About 40% of the income numbers are between 50k and 80k.

### transaction

There are 4 types of transcripts: normal transaction, offer received, offer viewed and offer complete. 

## Match different offer reponse behaviors

An offer will go through a funnel like this:

offer_received 
<br>|__ offer viewed 
<br>&nbsp; &nbsp; &nbsp; |__ offer completed
<br>&nbsp; &nbsp; &nbsp; |__ offer not completed
<br>|__ offer not viewed
<br>&nbsp; &nbsp; &nbsp; |__ offer completed
<br>&nbsp; &nbsp; &nbsp; |__ offer not completed

The transaction table looks like this:

![trans](https://raw.githubusercontent.com/tianyiwangnova/2020_project__Starbucks_Reward_Response/master/screenshot/transaction.png)

**Each person can receive the same type of offer for multiple times** and the offer id only tells what type of offer that is ---- it can't be used to recognize a unique offer. **Our challenge here is to match the various actions to tell if a certain offer was viewed or completed.** Also, customers might not receive all types of offers.

**We solve this problem by giving an order number of the transactions for each customer. The `offer id` plus the `rank` plus the `person id` can be seen as a "unique ID" for that offer.** We discovered that until the current offer expires, the app won't send another offer under the same type to that customer. We will attribute the earlist following actions (view the offer or complete the offer) to the earliest received offer (under the same offer id) with an extra condition that the following actions need to happen before that offer expires.

The chart below shows the logic of matching `offer_received` actions with `offer_viewed` actions:

*Assume that a customer has in total 8 offers related transactions*

![logic](https://raw.githubusercontent.com/tianyiwangnova/2020_project__Starbucks_Reward_Response/master/screenshot/logic.png)

We use this logic to match `offer-received` events with `offer-viewed` events and then match `offer-received` events and `offer-completed` events. Then we will match the 2 matched tables together.

After we matched the tables, we can attach the percentage numbers to the funnel:

offer_received 
<br>|__ offer viewed 74.16%
<br>&nbsp; &nbsp; &nbsp; |__ offer completed 35.9% 
<br>&nbsp; &nbsp; &nbsp; |__ offer not completed 38.26%
<br>|__ offer not viewed 25.84%
<br>&nbsp; &nbsp; &nbsp; |__ offer completed 7.5%
<br>&nbsp; &nbsp; &nbsp; |__ offer not completed 18.34%

3/4 of the offers can be viewed. More than 40% of the offers can be completed. The completion rate should be higher for just buy-one-get-one or discount offers though because informational offers won't need the customer to "complete". For buy-one-get-one offers and discount offers, the chance for an offer to be completed after the customer sees the offer is more than 50%. Discount offers are more likely to be completed when they are viewed but they will be less likely to be completed when they are not viewed (because they usually require higher purchase amount). 

![completetion](https://raw.githubusercontent.com/tianyiwangnova/2020_project__Starbucks_Reward_Response/master/screenshot/cr%20by%20offers.png)


## Exploration -- which demographic groups are more valuable?

Combining the demographic data with the transaction data, we can now visualize which demographic groups are more likely to complete offers and which groups have higher net revenues.

![completetion_rates](https://raw.githubusercontent.com/tianyiwangnova/2020_project__Starbucks_Reward_Response/master/screenshot/demo_and_completion_rates.png)


Female customers, older (> 35) customers, wealthier customers and customers who joined earlier are more valuable.

## Predictive modeling

In the notebook `02 Modeling`,  we fit `gradient boosting classifers` on the 4 subsets of data seperately:
* buy-one-get-one offers that were viewed
* buy-one-get-one offers that were not viewed
* discount offers that were viewed
* discount offers that were not viewed
<br>We used the feature importance matrixes to see which features are more useful when predicting whether the offers will be completed under difference circumstances.

Besides the demographic features and the promo features, we also built a few more features that describe the customers' `purchase related behaviors`, including:
<br>**offers_viewed_before:** at the time a person received an offer, how many offers they had seen before?
<br>**offers_completed_before:** at the time a person received an offer, how many offers they had completed before?
<br>**hours_since_last_viewed:** at the time a person received an offer, how many hours have passed since the person viewed the last offer
<br>**cum_amount:** at the time a person received an offer, how much money they had spent?

![balance](https://raw.githubusercontent.com/tianyiwangnova/2020_project__Starbucks_Reward_Response/master/screenshot/completion%20rates.png)
<br>In this project we are not facing data imbalance issue. Luckily there are no extremely small postive rates.

## Results

For buy-one-get-one offers that weren't viewed, `cumulative purchase amount at the time the customer received the offer`, `customer's income level`, `"customer_age" (how long has the customer been with us)` and `hours since the last offer was completed are important factors`. **Customers who have higher income levels and who have already had some purchases with us** are more likely to complete the offer.

Important features for buy-one-get-one offers that were viewed are similar to the un-viewed group, except that here gender plays a bigger role. **Compared to male customers, female customers are more likely to complete the offer when they know about it.**


For discount offers that weren't viewed, `cumulative purchase amount`, `customer's income level`, `whether this offer was sent through social channel` are important factors. It's interesting to see that **offers that were sent through social channels are more likely to be completed. (there are 2 such offers)**

For discount offers that were viewed, `customer age`, `cumulative purchase amount`, `gender`, `income level` and `time the offer was sent` are important factors. **We also see the pattern here that female customers are more likely to complete the offer when they have viewed the offer**. It seems that **customers who have completed some offers before the new offer are also more likely to complete the offer**.

# A senario to use the model...

Since there's no cost for an offer just to be sent --- if the customer doesn't complete the offer, we don't lose any money. One way for us to use the model to improve revenue is to **predict who will complete the offer without knowing it and we don't send offers to them**. This might not be a good idea because although we improve revenue in this way, it might hurt our relationship with the customers in the long run. It depends on the marketing strategies and if we are looking to cut the spending on promos, our models can help a lot.

We trained 2 models, one on the bogo offers that were not viewed and another one on the discount offers that were not viewed. Assume that we plan to send an offer potentially to everyone tomorrow but we want to exclude the customers who have higher chances to complete the offer without knowing it. So we plan to use the past data till 10 days before to train the model to make sure that in the training data all offers have expired.

![flowchart](https://raw.githubusercontent.com/tianyiwangnova/2020_project__Starbucks_Reward_Response/master/screenshot/predict.png)

In the notebook, we tried this process for the last recorded date in the data --- pretended that in that last day we wanted to optimize our campaign using the data we had collected. We trained the models with the data from the beginning till 10 days before the last day and tested the models on the offers received on the last day but weren't viewed till they expired. Our model for buy-one-get-one offers reaches an AUC score of 0.84 and our model for discount offers has an AUC score of 0.88.

![result](https://raw.githubusercontent.com/tianyiwangnova/2020_project__Starbucks_Reward_Response/master/screenshot/results.png)

