**Insights from Regression Model Analysis**:

1. **Model Performance Overview**:
   - Among the models tested, Polynomial Regression (Degree 2) performed better compared to Linear Regression, Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent, achieving an R² score of **0.3076**. This suggests that introducing non-linearity improved the model’s ability to capture more complex relationships in the data.
   - Linear Regression had a relatively low R² score of **0.1223**, indicating that it struggled to explain much of the variance in the target variable.
   - SGD Regression and Mini-Batch Gradient Descent exhibited **negative R² scores** (-0.7388 and -2.877, respectively), which indicates that these models were underperforming, potentially even worse than a naive mean-based prediction. This points to convergence issues or the need for hyperparameter tuning.

2. **Prediction vs Actual Performance**:
   - The **scatter plots** for all models showed a concentration of points around lower price values, with noticeable scatter and divergence from the perfect prediction line (dashed red). This pattern highlights the models' struggles to generalise well, particularly for higher-priced properties.
   - **Polynomial Regression** seemed to capture the variance slightly better than Linear Regression, showing fewer significant deviations from the perfect prediction line, especially for moderately priced properties.

3. **Residual Analysis**:
   - The **residual plots** indicate that there is significant heteroscedasticity across all models—residuals are not evenly distributed around zero, particularly for higher predicted values.
   - **Linear Regression** and **Polynomial Regression** both showed outliers with large residuals, indicating some properties are systematically over- or under-predicted, suggesting features that might be missing or require better transformations.
   - **SGD** and **Mini-Batch Gradient Descent** displayed large residual deviations, with Mini-Batch Gradient Descent, in particular, showing a highly uneven residual distribution. This unevenness indicates potential instability or inappropriate learning rates during training.

4. **Model Limitations and Recommendations**:
   - **Linear Regression's** limitations are evident, especially when predicting properties with more extreme prices, suggesting that the relationship between features and target may be highly non-linear.
   - **Polynomial Regression** showed improvements but at the cost of complexity and possible overfitting—particularly given the relatively modest R² score.
   - The poor performance of **SGD** and **Mini-Batch Gradient Descent** suggests that parameter tuning, such as learning rate or batch size adjustments, may be required. Moreover, feature scaling or further normalization techniques might improve their convergence and generalisation.

5. **Feature Importance and Next Steps**:
   - The general performance indicates that the features might not be sufficiently expressive to capture property value dynamics well. This calls for **feature engineering**—adding potentially useful features like property age, neighbourhood socio-economic indicators, or external amenities.
   - Additionally, using **regularisation techniques** such as Ridge or Lasso regression could help in improving model stability, particularly when handling multicollinearity or overfitting.

Overall, the analysis highlights the challenges in predicting property prices with high variability and complex underlying relationships. The insights gained from the scatter and residual plots can guide further model selection, feature engineering, and fine-tuning efforts to enhance predictive accuracy and generalisation.