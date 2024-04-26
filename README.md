# Welcome to LAR3S consulting
We undertake supermarket self-checkout anomaly detection.

# Objective
We want to identify fraudulent transactions. 
We can reasonably assume that ~±15% are fraudulent.

- We can put in checker 150/1,000
- Our test run gave us 13%

Our objective naturally follows that we find some kind of features of splitting/clustering/sorting dataset in such a way that we come up with group of receipts that are higher/lower than 15%. 

## How would we do that?
We can find patterns in data, and see if our pattern would give us a good score on Checker. 

- Summary statistics (Mean, median, max...) of prices/departments/time
- Some derived features from existing data
  - Frequency of changing the department (High hope)
  - Association rules (Less hope)
- Reducing data and sort 
  - It might be the case we do not have to consider all scans per receipt
  - Maybe average time/SD per time per receipt tells us a good enough story?
 
## How would we justify that?
We can do exploratory data analysis, and bring attention to some unusual characteristics. 
Then link the abnormal behavior to the original problem, and prescribe actionable plans to reduce fraud cases for AHA.

## What should be in EDA?
Here are some suggestions. 
  - How much customers spend (time/money)
  - How do customers visit (department)
  - Start bringing attention to abnormality (We don't know what this would be yet)

## Should we use machine learning? 
Yes we can, then the task becomes unsupervised classification. 
  - Why unsupervised? There is no true label provided
  - Why classification? Our output per receipt should be classified fraud/or not.

## What are some good unsupervised classification algorithms?
We can try K means, DBSCAN, hierarchical clustering, and so on. 

## What will be the useful splits/clusters? 
We don't know, but we can test our splits on Checker. Again, our goal is to identify top 150 per 1000, not top 10. So very specific clusters may not be that useful in our case. 

## Should we include many splits/clusters? 
Yes and no. We need the suspicious group to have at least 150 observations so we can run it on checker, so 50/50 split would not be very useful.
If we include too many splits, we might be overfitting. 

## How would we justify our suggestions of splits/clusters?
Linking to EDA, we can show that this characteristic has higher chance of being sus. Maybe recall our year 1 period 6 class Industry Analytical Reporting and attach some informative graphics and notations.

## How can we improve our model?
We first need to find some clustering model that would give us reasonably good fraud cases (20%+). Then we can add features on top of it to improve our numbers.

## What are some interesting features we can consider? 
Here are some suggestions, but the more important consideration is that our cluster can be ONLY evaluated through checker, and how good a feature sounds has less to do with actually checking its performance. The most interesting feature might not give us enough fraudulent receipts, which then we cannot really test our hypotheses. 

 - Time-based
  - Total duration of the visit (difference between the first and last scan times)
  - Average time between scans
  - Variance or standard deviation of time between scans
  - Minimum and maximum time gaps between consecutive scans
  - Proportion of time spent in each department (based on the cumulative time between scans in each department)

- Price-based
  - Average price of scanned items
  - Variance or standard deviation of prices
  - Minimum and maximum prices of scanned items
  - Proportion of total cost spent in each department

- Department-level
  - Number of unique departments visited
  - Distribution of scans across departments (e.g., entropy or Gini coefficient)
  - Sequence or order of department visits

- Path based 
  - Number of "back-and-forth" movements between departments (changing departments multiple times)
  - Patterns or sequences of department transitions
  - Unusual or unexpected department transitions (based on frequent patterns)

## We found some solid set of features and they return 30% fraud rate, what do we do?
We can start making slides and prescribe some recommendations for AHA.
One nice way to structure:
  - Situation
  - Complication
  - Question

## We are having trouble to find anything that definitely surpasses baseline of 15%, what do we do?
We might be overcomplicating the task. Maybe consider statistics-based methods then machine learning. 

## How should we write reports?
- Lead with a (one-page) executive summary
- Use “pyramid principle” to slowly build up, so the reader can easily skip the technical details
- Be as concise as possible while still communicating effectively

