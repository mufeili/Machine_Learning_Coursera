%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = [0.01 0.03 0.1 0.3];
num_iters = 400;

% Init Theta and Run Gradient Descent with different alphas
theta = zeros(3,1);
[theta1, J_history1] = gradientDescentMulti(X, y, theta, alpha(1), num_iters);
theta = zeros(3, 1);
[theta2, J_history2] = gradientDescentMulti(X, y, theta, alpha(2), num_iters);
theta = zeros(3, 1);
[theta3, J_history3] = gradientDescentMulti(X, y, theta, alpha(3), num_iters);
theta = zeros(3, 1);
[theta4, J_history4] = gradientDescentMulti(X, y, theta, alpha(4), num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history1), J_history1, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
hold on;
plot(1:numel(J_history2), J_history2, '-r', 'LineWidth', 2);
plot(1:numel(J_history3), J_history3, '-k', 'LineWidth', 2);
plot(1:numel(J_history4), J_history4, '-g', 'LineWidth', 2);
legend('alpha = 0.01', 'alpha = 0.03', 'alpha = 0.1', 'alpha = 0.3');

% Display gradient descent's result
fprintf('\n');
fprintf('Theta computed from gradient descent with alpha = 0.01: \n');
fprintf(' %f \n', theta1);
fprintf('\n');
fprintf('Theta computed from gradient descent with alpha = 0.03: \n');
fprintf(' %f \n', theta2);
fprintf('\n');
fprintf('Theta computed from gradient descent with alpha = 0.1: \n');
fprintf(' %f \n', theta3);
fprintf('\n');
fprintf('Theta computed from gradient descent with alpha = 0.3: \n');
fprintf(' %f \n', theta4);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
fprintf('\n');
sample = [1, (1650-mu(1))/sigma(1), (3-mu(2))/sigma(2)]
PRICE1 = sample * theta1
PRICE2 = sample * theta2
PRICE3 = sample * theta3
PRICE4 = sample * theta4
% ============================================================
fprintf('\n');
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'...
         'with alpha = 0.01'], PRICE1);

fprintf('\n');
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'...
         'with alpha = 0.03'], PRICE2);
        
fprintf('\n');
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'...
         'with alpha = 0.1'], PRICE3);
        
fprintf('\n');
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'...
         'with alpha = 0.3'], PRICE4);
        
fprintf('\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = [1,1650,3] * theta
difference1 = price - PRICE1
difference2 = price - PRICE2
difference3 = price - PRICE3
difference4 = price - PRICE4
% ============================================================
fprintf('\n');
        
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);
        
fprintf('\n');

fprintf(['Difference between prediced prices of a 1650' ...
         'sq-ft, 3 br house (using normal equations and'...
         'gradient descent separately with alpha = 0.01'...
         '):\n $%f\n'], difference1);
        
fprintf('\n');

fprintf(['Difference between prediced prices of a 1650' ...
         'sq-ft, 3 br house (using normal equations and'...
         'gradient descent separately with alpha = 0.03'...
         '):\n $%f\n'], difference2);
        
fprintf('\n');

fprintf(['Difference between prediced prices of a 1650' ...
         'sq-ft, 3 br house (using normal equations and'...
         'gradient descent separately with alpha = 0.1)'...
         ':\n $%f\n'], difference3);
        
fprintf('\n');

fprintf(['Difference between prediced prices of a 1650' ...
         'sq-ft, 3 br house (using normal equations and'...
         'gradient descent separately with alpha = 0.3)'...
         ':\n $%f\n'], difference4);
