## Telestra Network

_This repository contains code regarding Kaggle's Telestra Network competition._

### Analysis
We are looking here at event data. Which means that for each event (a single prediction), we have many rows of training data. This means that we have a one to many relationship for X_test to y_test.
We have two options:
1. Treat each row independently and try to predict that row's event and then "vote" on the final one.
   - predict many --> vote on final
2. Aggregate the data for that particular event and then predict on a single row.
   - get aggregate data --> predict single row

_A third option would be to possibly combine the two approaches. Both can be used to vote on a final outcome._

**I chose option 2, so let's see how the data for these were constructed.**

_Note: At the time of this competition, many of sklearn's modeling functions needed dummy variables to work, that is why we had to use the get_dummies function and create many columns. I believe that is no longer entirely necessary_
