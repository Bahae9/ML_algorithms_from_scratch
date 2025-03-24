import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, lr=0.01, tolerance=1e-16, verbose=True, normalization_method='MinMax'):
        self.lr = lr  # Learning rate
        self.tolerance = tolerance  # Stopping condition
        self.verbose = verbose  # Print progress
        self.w = None
        self.b = 0.0
        self.loss_history = []
        self.normalization_method = normalization_method
        self.scaling_applied = False
    
    '''
        ToDo:
            - Implementing dynamic learning rate based on the gradient's behavior.
    '''
    @staticmethod
    def info():
        return '''
            LinearRegression Class:
            -----------------------
            A simple implementation of Linear Regression with optional normalization and multiple optimizers.

            Parameters:
            - lr (float): Learning rate for gradient descent. Default is 0.01.
            - tolerance (float): Threshold for early stopping. Default is 1e-16.
            - verbose (bool): If True, prints training progress. Default is True.
            - normalization_method (str): Type of normalization. Options: 'MinMax' (default) or 'Std' (Standardization).

            Methods:
            --------
            - fit(X, y, epochs=10000, optimizer='BGD'):
                Trains the model using the selected optimizer.
                - X: Feature matrix (numpy array or list).
                - y: Target values (numpy array or list).
                - epochs: Number of training iterations (default: 10000).
                - optimizer: Optimization method ('BGD' for Batch Gradient Descent, 'SGD' for Stochastic Gradient Descent, 'man' for manual weight setting).

            - predict(X_test):
                Predicts output for given test data.
                - X_test: Input features.

            - score(y_true, y_pred):
                Evaluates the model's performance using:
                - R² Score: Measures variance explanation.
                - RMSE: Measures the root mean squared error.

            - plot_loss():
                Plots the loss function over iterations.

            - plot_regression_line(X, y):
                Visualizes the fitted regression line on 2D data.

            Normalization Adjustments:
            --------------------------
            - If 'MinMax' normalization is applied, weights (w) and bias (b) are rescaled back to original scale.
            - If 'Std' standardization is used, weights and bias are also adjusted accordingly.

            Example Usage:
            --------------
            model = LinearRegression(lr=0.01, normalization_method='Std')
            model.fit(X_train, y_train, optimizer='BGD')
            predictions = model.predict(X_test)
            performance = model.score(y_test, predictions)
            model.plot_loss()
            model.plot_regression_line(X_train, y_train)

               '''
    
    def _normalize(self, X: np.ndarray, y: np.ndarray):
        """Applies Min-Max scaling or Standardization."""
        if self.normalization_method == 'MinMax':
            self.X_min, self.X_max = np.min(X, axis=0), np.max(X, axis=0)
            self.y_min, self.y_max = np.min(y), np.max(y)
            denomX = self.X_max - self.X_min
            denomX[denomX == 0] = 1
            X = (X - self.X_min) / denomX
            denomY = self.y_max - self.y_min
            if denomY == 0:
                denomY = 1
            y = (y - self.y_min) / denomY
        elif self.normalization_method == 'Std':
            self.X_mean, self.X_std = np.mean(X, axis=0), np.std(X, axis=0)
            self.y_mean, self.y_std = np.mean(y), np.std(y)
            self.X_std[self.X_std == 0] = 1
            X = (X - self.X_mean) / self.X_std
            if self.y_std == 0:
                self.y_std = 1
            y = (y - self.y_mean) / self.y_std
        else:
            raise ValueError("Invalid normalization method. Use 'MinMax' or 'Std'.")
        return X, y
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10000, optimizer: str = 'BGD'):
        X, y = np.array(X), np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Convert (m,) → (m,1)

        m, n = X.shape
        self.w = np.zeros(n)  # Initialize weights

        if np.abs(X).max() > 1e6 or np.abs(y).max() > 1e6:
            print("Large values detected! Applying scaling to stabilize training.")

            X, y = self._normalize(X, y)
            self.scaling_applied = True

        if optimizer == 'BGD':
            self.batch_gradient_descent(X, y, epochs)
        elif optimizer == 'SGD':
            self.stochastic_gradient_descent(X, y, epochs)
        else:
            raise ValueError(
                "Error: Optimizer does not exist. Use LinearRegression.info() for help."
                )

        if self.scaling_applied:
            # Adjusting weight and bias
            if self.normalization_method == 'MinMax':
                self.w = self.w * (self.y_max - self.y_min) / (self.X_max - self.X_min)
                self.b = - np.dot(self.w, self.X_min) + self.b * (self.y_max - self.y_min) + self.y_min
            
            elif self.normalization_method == 'Std':
                self.w = self.w * (self.y_std / self.X_std)
                self.b = self.b * self.y_std + self.y_mean - np.dot(self.w, self.X_mean)

        print("w:", self.w, "b:", self.b)

    def batch_gradient_descent(self, X, y, epochs):
        m = len(X)
        prev_loss = float('inf')  # Track loss for early stopping

        for i in range(epochs):
            y_pred = np.dot(X, self.w) + self.b
            loss = self.mean_squared_error(y, y_pred)
            self.loss_history.append(loss)  # Store loss for plotting

            # Compute gradients
            dw = (1 / m) * np.dot(X.T, (y_pred - y))  # Partial derivative w.r.t. w
            db = (1 / m) * np.sum(y_pred - y)  # Partial derivative w.r.t. b

            # Update weights and bias
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Early stopping condition
            if np.abs(prev_loss - loss) < self.tolerance:
                if self.verbose:
                    print(f"Early stopping at iteration {i}, Loss: {loss:.6f}")
                break

            prev_loss = loss

            # Print progress every 50 iterations
            if self.verbose and i % 50 == 0:
                print(
                    f"Iteration {i}: Loss = {loss:.6f}, w = {self.w}, b = {self.b}")

    def stochastic_gradient_descent(self, X, y, epochs):
        m = len(X)
        prev_loss = float('inf')

        for epoch in range(epochs):
            for i in range(m):
                xi = X[i, :].reshape(1, -1)  # Ensure correct shape
                yi = y[i]

                y_pred = np.dot(xi, self.w) + self.b
                loss = self.mean_squared_error(
                    np.array([yi]), np.array([y_pred]))

                # Store only final loss of each epoch for visualization
                if i == m - 1:
                    self.loss_history.append(loss)

                # Compute gradients
                dw = np.dot(xi.T, (y_pred - yi))
                db = (y_pred - yi)

                # Update parameters
                self.w -= self.lr * dw.flatten()
                self.b -= self.lr * db

            # Compute loss after full epoch
            total_loss = self.mean_squared_error(y, np.dot(X, self.w) + self.b)

            if np.abs(prev_loss - total_loss) < self.tolerance:
                if self.verbose:
                    print(
                        f"Early stopping at epoch {epoch}, Loss: {total_loss:.6f}")
                break

            prev_loss = total_loss

            # Print progress every 50 epochs
            if self.verbose and epoch % 50 == 0:
                print(
                    f"Epoch {epoch}: Loss = {total_loss:.6f}, w = {self.w}, b = {self.b}")

    def mean_squared_error(self, y_true, y_pred):
        m = len(y_true)
        return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)

    def predict(self, X_test):
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        return np.dot(X_test, self.w) + self.b

    def score(self, y_true, y_pred):
        """
        Computes:
        - R² Score (how well the model explains variance)
        - RMSE (how large the typical error is)
        """
        # R² Score
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)

        # RMSE (Root Mean Squared Error)
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        return {"R2 Score": r2, "RMSE": rmse}

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss over Iterations")
        plt.show()

    def plot_regression_line(self, X, y):
        plt.scatter(X, y, color="blue", label="Actual data")
        y_pred = self.predict(X)
        plt.plot(X, y_pred, color="red", label="Regression line")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Linear Regression Fit")
        plt.legend()
        plt.show()
