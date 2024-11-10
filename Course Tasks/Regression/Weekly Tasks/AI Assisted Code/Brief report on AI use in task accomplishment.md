# Record of AI-Assisted and User-Contributed Code Snippets and Ideas

## 5. AI Tools and Resources Used
Throughout this project, I utilized various AI tools and resources to assist with code development, error checking, and visualization. The following AI tools were employed:
- **Cursor AI**: Used for general AI edits and code suggestions.
- **ChatGPT O1 Preview*: Served as my main AI tutor, providing explanations and guidance on machine learning concepts and implementation.
- **ChatGPT 4.0 with Canvas**: Utilized for full codebase updates, helping to refine code I initially wrote myself by removing typing errors and other minor issues in the process.
- **Claude AI**: Employed to create mock interfaces, helping to visualize the code and understand what is actually happening as we tweak weights, optimize cost functions, and make other adjustments.

The main reference book used throughout this project was:
Géron, A., 2019. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: concepts, tools, and techniques to build intelligent systems. 2nd ed. Sebastopol, CA: O'Reilly Media.
OpenAI, 2023. ChatGPT (GPT-4) [Large language model]. Available at: <https://chat.openai.com/chat> [Accessed 2 April 2024].
Anthropic, 2023. Claude AI [Large language model]. Available at: <https://www.anthropic.com> [Accessed 2 April 2024].

This comprehensive resources provided invaluable insights into machine learning concepts, tools, and techniques, serving as a foundation for much of the work completed.

## 6. Acknowledgement of AI Assistance
I acknowledge that while the writing and implementation are my own, and I take full responsibility for any errors, the aforementioned AI tools were instrumental in various aspects of this project. They assisted in generating initial section structures, providing code suggestions, and helping to visualize complex concepts. The use of these tools has significantly enhanced my understanding of machine learning principles and improved the overall quality of the project.

For more specific details on AI usage, there is a dedicated folder documenting the instances where AI was employed, including the relevant code it generated. Additionally, I have commented directly on this file to indicate the parts that are fully AI-generated.



## Introduction

During the development of the exploratory data analysis (EDA) code, I collaborated extensively with the AI assistant to improve the code's functionality, readability, and effectiveness. This document summarizes the AI-assisted code snippets and ideas, as well as my own contributions, highlighting how they were incorporated into the main task implementation.

## 1. My Suggestions on Handling Skewed Data in Plots

I identified that the plots, especially the distribution of house prices, were highly skewed with most values concentrated near the lower end and extreme outliers stretching the x-axis. I suggested several improvements:

- **Logarithmic Transformation**: I proposed applying a logarithmic transformation to the price data to normalize the distribution.
- **Capping Extreme Values**: I recommended capping extreme values at a certain threshold (e.g., the 95th percentile) to remove outliers.
- **Using Smaller Bin Sizes**: I suggested using smaller bin sizes to provide finer resolution of the distribution.

These suggestions were incorporated into the code:

- Applied Logarithmic Transformation: The code now uses `np.log1p()` to apply a log transformation to the 'Price ($)' column and plots the log-transformed distribution to achieve a more normal distribution and better visualization.
- Capped Extreme Values: The 'Price ($)' values are now filtered to include only those below the 95th percentile, and the capped distribution is plotted to reduce the impact of extreme outliers.
- Adjusted Bin Sizes: The number of bins in histograms has been modified to enhance granularity and interpretability.

## 2. My Identification of Issues with 'Price Sqr Ft ($/sqft)' Column

I noticed that the plot of "Price per Square Foot" was not displaying any data. I suspected that the 'Price Sqr Ft ($/sqft)' column contained non-numeric data or NaN values causing the plot to fail. I suggested checking for non-numeric or missing data in the column, cleaning the data by converting the column to numeric and handling errors appropriately. I provided sample data showing values like '260/', indicating the presence of non-numeric characters.

These suggestions were incorporated into the code:

- Data Cleaning: The code now removes trailing '/' characters using `df[col].astype(str).str.replace('/', '', regex=False)` and converts the cleaned column to numeric using `pd.to_numeric(df[col], errors='coerce')`.
- Data Filtering: Rows with NaN values are dropped, and zero or negative values are filtered out.
- Plotting with Cleaned Data: The distribution is now replotted with the cleaned data, ensuring the plot displays correctly.

## 3. My Request to Apply Best Methods for Improving 'Price per Square Foot' Plot

I described that the distribution plot for "Price per Square Foot" was highly skewed, with most data points at the low end and a few extremely high values stretching the x-axis. I suggested several solutions:

- Apply Logarithmic Transformation: I recommended applying a log transformation to compress extreme values.
- Remove or Cap Outliers: I proposed capping values at the 95th percentile to focus on the bulk of the data.
- Zoom into Specific Price Ranges: I suggested zooming into specific price ranges by filtering the data.

These suggestions were incorporated into the code:

- Capped Outliers and Applied Log Transformation: The 'Price Sqr Ft ($/sqft)' values are now capped at the 95th percentile and a log transformation is applied to the capped data.
- Plotted Enhanced Distribution: The code now plots the log-transformed, capped distribution to improve interpretability and provides alternative plots without log transformation for comparison.

## 4. My Feedback on Enhancing Plot Interpretability and Coding Standards

I provided feedback on improving visualization, refining pair plots, adding visual aids, and adhering to coding standards:

- Improving Visualization: I noted that the visualization code did not consider enhancing interpretability and suggested adding detailed titles, axis labels, legends, and color schemes.
- Refining Pair Plots: I observed that the pair plot used all selected numeric columns without justification and recommended performing correlation analysis to identify significant relationships and focus on plotting the most relevant features.
- Adding Visual Aids: I pointed out the lack of visual aids such as vertical/horizontal lines indicating thresholds in plots.
- Adhering to Coding Standards: I highlighted inconsistencies in comments, spacing, and block indentation, and suggested using f-strings for printing messages for cleaner and more Pythonic code.

These suggestions were incorporated into the code:

- Enhanced Plot Details: The code now includes more descriptive titles, axis labels, and legends with appropriate font sizes, and applies consistent color schemes with improved overall aesthetics.
- Performed Correlation Analysis: A correlation matrix is now calculated to identify features most correlated with 'Log Price ($)', and pair plots focus on the top correlated features.
- Included Visual Aids: Vertical lines indicating mean values in histograms and regression lines in scatter plots have been added to visualize trends.
- Improved Coding Standards: The code now ensures consistent use of comments, spacing, and indentation, updates print statements to use f-strings, and adds docstrings and comments for better code documentation.

## 5. AI Assistant's Suggestions and Code Enhancements

The AI assistant provided several suggestions and code enhancements:

- Multiprocessing Issues: The AI identified that the code was running too long and not producing plots, possibly due to conflicts between multiprocessing and plotting libraries. It suggested removing multiprocessing to simplify execution and avoid plotting conflicts.
- Plotting Backend Issues: The AI recommended adjusting the matplotlib backend or saving plots to files if running in a non-GUI environment.
- Data Sampling for Large Plots: It proposed sampling data for computationally intensive plots like pair plots to speed up rendering and reduce execution time.
- Adding Regression Lines to Scatter Plots: The AI suggested adding regression lines to scatter plots to visualize trends and relationships.
- Setting Seaborn Styles and Color Palettes: It advised setting a consistent Seaborn style and muted color palette for all plots.
- Error Handling and Warnings: The AI recommended including error handling for potential exceptions and adding warnings when expected columns were not found.

These suggestions were incorporated into the code as described in the previous sections.

## 6. Collaborative Enhancements to Coding Standards

Both the AI assistant and I emphasized the importance of following coding standards. We ensured consistent indentation, spacing, and commenting throughout the code, making it cleaner, more maintainable, and aligned with best practices.

## Integration into Main Task Implementation

The collaboration between me and the AI assistant resulted in a comprehensive and effective EDA codebase. My domain knowledge and critical observations about the data guided the improvements, while the AI assistant provided technical solutions and best practices to implement them.

The main areas where contributions were integrated include data cleaning and preprocessing, data visualization, statistical analysis, and coding practices.

## Distinguishing Contributions

My contributions focused on problem identification, practical solutions, code quality, and providing data samples and insights. The AI assistant's contributions centered on more technical implementation, best practices, and problem-solving and complicated error handling.

## Reflection

The collaborative effort between me and the AI assistant led to a robust and polished EDA script. My insights into the data and proactive suggestions significantly improved the analysis. The AI assistant's technical expertise facilitated the implementation of these ideas and contributed additional enhancements.