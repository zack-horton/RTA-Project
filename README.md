# RTA-Project
15.S07: Real-time Analytics for Digital Platforms Project

Link to [Kaggle for datafile](https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma)

**Description:**

The primary goal of the project was to develop a strategy-based application to help riders and drivers of the popular rideshare apps from Uber and Lyft. 

**Rider Services:**

For riders, we applied 8 different machine learning models (OLS, Ridge, LASSO, CART (regression), RF, XGBoost, NN, DNN) to both Uber and Lyft rides in the Boston, MA area. We wanted to learn the pricing strategies of each company so we could compare them across the two different services. After receiving ride information from the user, we estimate the cost of the ride and recommend the cheaper service provider, and also give a 95% confidence interval for the estimation. We also provide a model-by-model breakdown of each estimation for both companies, totaling 16 different data points for a given ride that the user can compare.

**Driver Services:**

For drivers, we applied Q-Learning (a form of reinforcement learning) to understand what would be the optimal decision for a driver in a current area of Boston, MA (assuming they have multiple options for the next ride they will service). This would allow drivers to be able to understand and act on forward-looking information, optimizing their long-run reward based on their current situation.
