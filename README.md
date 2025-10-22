# Linear Regression with Gradient Descent

This project uses a dataset of study hours (`Hours`) and corresponding exam scores (`Scores`) to perform linear regression. The dataset is loaded from a CSV file (`data.csv`), and a simple linear model is trained using gradient descent to fit the best line to the data. The goal is to predict exam scores based on hours studied by optimizing the slope (`m`) and intercept (`b`) of the line.

The implementation begins by loading the dataset and initializing the model with arbitrary values for `m` and `b`. The gradient descent algorithm iterates over multiple epochs, adjusting the parameters to minimize the Mean Squared Error (MSE) between the predicted and actual scores. After training, the model’s performance is visualized with two plots: one showing the decrease in MSE over epochs and another displaying the best-fit line on top of the original data points.

The results show that after training, the model parameters converge to values that minimize the MSE, indicating that the model has successfully learned the relationship between hours studied and exam scores. The generated plots demonstrate a clear reduction in error as the model trains, and the best-fit line closely matches the pattern of the data, confirming the effectiveness of the gradient descent approach.
