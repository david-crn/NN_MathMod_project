import numpy as np
import matplotlib.pyplot as plt


class Optimizer:
    def __init__(self, function, learning_rate=0.005, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10000, momentum = 0.9,
                 patience=10, delta=1e-4):
        
        self.function = function  # User-defined function
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.momentum = momentum
        self.patience = patience
        self.delta = delta


    def early_stopping(self, best_loss, loss, wait):
        """Checks if early stopping criteria are met."""
        norm = np.linalg.norm(best_loss - loss)

        if norm > self.delta:
            return loss, 0  # Improvement detected, reset wait counter
        else:
            return best_loss, wait + 1  # No improvement, increment wait counter
    

    def adam(self, init_params):
        params = np.array(init_params, dtype=np.float64)
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        trajectory = [params.copy()]

        best_loss = float('inf')
        wait = 0
        
        for t in range(1, self.num_iterations + 1):
            # Add gradient noise
            sigma_t = (self.learning_rate / (1 + t) ** self.momentum)
            grad = np.array(self.function(params))
            grad += np.random.normal(0, sigma_t, size=grad.shape)

            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            loss = self.function(params)

            # Early stopping check
            best_loss, wait = self.early_stopping(best_loss, loss, wait)
            if wait >= self.patience:
                print(f"Early stopping at iteration {t}")
                break

            trajectory.append(params.copy())
        
        return params, np.array(trajectory)
    

    def gd(self, init_params):
        params = np.array(init_params, dtype=np.float64)
        trajectory = [params.copy()]

        best_loss = float('inf')
        wait = 0
        
        for t in range(self.num_iterations):
            grad = np.array(self.function(params))
            params -= self.learning_rate * grad
            loss = self.function(params)

            # Early stopping check
            best_loss, wait = self.early_stopping(best_loss, loss, wait)
            if wait >= self.patience:
                print(f"Early stopping at iteration {t}")
                break

            trajectory.append(params.copy())
        
        return params, np.array(trajectory)
    

    def sgd(self, init_params, batch_size=1):
        params = np.array(init_params, dtype=np.float64)
        trajectory = [params.copy()]

        best_loss = float('inf')
        wait = 0

        for t in range(self.num_iterations):
            # Generate 10 perturbations near params
            perturbations = np.random.normal(0, 0.1, size=(batch_size, *params.shape))

            # Compute gradients for all perturbations and average them
            grads = np.array([self.function(params + perturbations[idx]) for idx in range(batch_size)])
            grad = np.mean(grads, axis=0)  # Average over 10 samples

            # Update parameters using averaged gradient
            params -= self.learning_rate * grad
            loss = self.function(params)

            # Early stopping check
            best_loss, wait = self.early_stopping(best_loss, loss, wait)
            if wait >= self.patience:
                print(f"Early stopping at iteration {t}")
                break

            trajectory.append(params.copy())

        return params, np.array(trajectory)
    

    def sgdm(self, init_params):
        params = np.array(init_params, dtype=np.float64)
        velocity = np.zeros_like(params)
        trajectory = [params.copy()]

        best_loss = float('inf')
        wait = 0

        for t in range(self.num_iterations):
            # Add gradient noise
            sigma_t = (self.learning_rate / (1 + t) ** self.momentum)
            grad = np.array(self.function(params))
            grad += np.random.normal(0, sigma_t, size=grad.shape)

            # Apply SGDM update rule with momentum
            velocity = self.momentum * velocity + (1 - self.momentum) * grad
            params -= self.learning_rate * velocity
            loss = self.function(params)

            # Early stopping check
            best_loss, wait = self.early_stopping(best_loss, loss, wait)
            if wait >= self.patience:
                print(f"Early stopping at iteration {t}")
                break

            trajectory.append(params.copy())

        return params, np.array(trajectory)
    

    def nesterov(self, init_params):
        params = np.array(init_params, dtype=np.float64)
        velocity = np.zeros_like(params)
        trajectory = [params.copy()]

        best_loss = float('inf')
        wait = 0

        for t in range(self.num_iterations):
            # Lookahead step
            lookahead_params = params + self.momentum * velocity
            
            # Compute gradient at lookahead position
            sigma_t = (self.learning_rate / (1 + t) ** self.momentum)
            grad = self.function(lookahead_params)
            grad += np.random.normal(0, sigma_t, size=grad.shape)
            
            # Apply Nesterov update rule
            velocity = self.momentum * velocity - self.learning_rate * grad
            params += velocity
            loss = self.function(params)

            # Early stopping check
            best_loss, wait = self.early_stopping(best_loss, loss, wait)
            if wait >= self.patience:
                print(f"Early stopping at iteration {t}")
                break

            trajectory.append(params.copy())

        return params, np.array(trajectory)
    

    def adagrad(self, init_params):
        params = np.array(init_params, dtype=np.float64)
        grad_squared_accum = np.zeros_like(params)
        trajectory = [params.copy()]

        best_loss = float('inf')
        wait = 0
        
        for t in range(self.num_iterations):
            grad = np.array(self.function(params))
            grad_squared_accum += grad ** 2
            adjusted_lr = self.learning_rate / (np.sqrt(grad_squared_accum + self.epsilon))
            params -= adjusted_lr * grad
            loss = self.function(params)

            # Early stopping check
            best_loss, wait = self.early_stopping(best_loss, loss, wait)
            if wait >= self.patience:
                print(f"Early stopping at iteration {t}")
                break

            trajectory.append(params.copy())
        
        return params, np.array(trajectory)
    

    def adadelta(self, init_params):
        params = np.array(init_params, dtype=np.float64)
        E_grad_sq = np.zeros_like(params)
        E_dx_sq = np.zeros_like(params)
        trajectory = [params.copy()]

        best_loss = float('inf')
        wait = 0
        
        for t in range(self.num_iterations):
            grad = np.array(self.function(params))
            E_grad_sq = self.beta1 * E_grad_sq + (1 - self.beta1) * (grad ** 2)
            dx = - (np.sqrt(E_dx_sq + self.epsilon) / np.sqrt(E_grad_sq + self.epsilon)) * grad
            E_dx_sq = self.beta1 * E_dx_sq + (1 - self.beta1) * (dx ** 2)
            params += dx
            loss = self.function(params)

            # Early stopping check
            best_loss, wait = self.early_stopping(best_loss, loss, wait)
            if wait >= self.patience:
                print(f"Early stopping at iteration {t}")
                break

            trajectory.append(params.copy())
        
        return params, np.array(trajectory)
    

    def adamax(self, init_params):
        params = np.array(init_params, dtype=np.float64)
        m = np.zeros_like(params)
        u = np.zeros_like(params)
        trajectory = [params.copy()]

        best_loss = float('inf')
        wait = 0
        
        for t in range(1, self.num_iterations + 1):
            grad = np.array(self.function(params))
            m = self.beta1 * m + (1 - self.beta1) * grad
            u = np.maximum(self.beta2 * u, np.abs(grad))
            params -= (self.learning_rate / (u + self.epsilon)) * m
            loss = self.function(params)

            # Early stopping check
            best_loss, wait = self.early_stopping(best_loss, loss, wait)
            if wait >= self.patience:
                print(f"Early stopping at iteration {t}")
                break

            trajectory.append(params.copy())
        
        return params, np.array(trajectory)
    

    def nadam(self, init_params):
        params = np.array(init_params, dtype=np.float64)
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        trajectory = [params.copy()]

        best_loss = float('inf')
        wait = 0
        
        for t in range(1, self.num_iterations + 1):
            grad = np.array(self.function(params))
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            m_hat = (self.beta1 * m) / (1 - self.beta1 ** t) + ((1 - self.beta1) * grad) / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            loss = self.function(params)

            # Early stopping check
            best_loss, wait = self.early_stopping(best_loss, loss, wait)
            if wait >= self.patience:
                print(f"Early stopping at iteration {t}")
                break

            trajectory.append(params.copy())

        return params, np.array(trajectory)