%% PINN for a Mass-Spring-Damper System (Underdamped Oscillator)

clear; clc; close all;

%% Problem Setup and Data Generation

% Parameters of the oscillator
d  = 2;      % damping factor
w0 = 20;     % natural frequency
mu = 2*d;    % damping coefficient
k  = w0^2;   % stiffness coefficient

% Analytical solution parameters for underdamped oscillator
%   y(t) = A*exp(-d*t)*cos(w*t + phi)
w   = sqrt(w0^2 - d^2);
phi = atan(-d / w);
A   = 1/(2*cos(phi));

% Define the analytical solution as an anonymous function
oscillator = @(x) exp(-d*x) .* (2*A*cos(phi + w*x));

% Generate full domain data for plotting (fine grid)
numPoints = 500;
x = linspace(0,1,numPoints)';   % column vector
y = oscillator(x);

% Select training data from the left part of the domain (10 points)
indices = 1:20:200; 
x_data = x(indices);
y_data = y(indices);

% Plot the exact solution and training data
figure;
plot(x, y, 'k-', 'LineWidth', 2); hold on;
scatter(x_data, y_data, 80, 'ro', 'filled');
xlabel('x'); ylabel('y');
title('Exact solution and training data');
legend('Exact','Training','Location','best');
hold off;

%% Convert Data to dlarray for Differentiation
x_data_dl = dlarray(x_data','CB');  % shape = 1 x N
y_data_dl = dlarray(y_data','CB');

% Create physics collocation points: 30 points in [0,1]
x_phys = linspace(0,1,30)';
x_phys_dl = dlarray(x_phys','CB');  % shape = 1 x 30

%% Define the Neural Network Architecture
layers = [
    featureInputLayer(1, 'Normalization','none','Name','input')
    fullyConnectedLayer(32, 'Name','fc1')
    tanhLayer('Name','tanh1')
    fullyConnectedLayer(32, 'Name','fc2')
    tanhLayer('Name','tanh2')
    fullyConnectedLayer(32, 'Name','fc3')
    tanhLayer('Name','tanh3')
    fullyConnectedLayer(1, 'Name','output')];

lgraph = layerGraph(layers);
net = dlnetwork(lgraph);

% If you're on R2023a or later, you *could* do:
%   net.Acceleration = 'none';
% but it is not needed (and not recognized) in older versions.

%% Training Setup
numIterations = 10000;
learningRate  = 1e-4;

trailingAvg   = [];
trailingAvgSq = [];
lossHistory   = zeros(numIterations, 1);

%% PINN Training Loop
figure;
for iter = 1:numIterations
    
    % Evaluate model loss and its gradients via dlfeval
    [lossVal, gradients, lossDataVal, lossPhysVal] = ...
        dlfeval(@modelLoss, net, x_data_dl, y_data_dl, x_phys_dl, mu, k);
    
    lossHistory(iter) = gather(extractdata(lossVal));
    
    % Update network parameters using the ADAM optimizer
    [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
        trailingAvg, trailingAvgSq, iter, learningRate);
    
    % Display and plot every 150 iterations
    if mod(iter,150) == 0
        % Evaluate the network on the full domain
        x_full_dl = dlarray(x','CB');
        y_pred_dl = forward(net, x_full_dl, 'Outputs','output');
        y_pred    = gather(extractdata(y_pred_dl))';  % make it column vector
        
        % Plot the results
        subplot(1,2,1);
        plot(x, y, 'k-', 'LineWidth',2); hold on;
        scatter(x_data, y_data, 80, 'ro', 'filled');
        plot(x, y_pred, 'b--', 'LineWidth',2);
        xlabel('x'); ylabel('y');
        title(sprintf('Iteration %d, Loss=%.3e', iter, lossHistory(iter)));
        legend('Exact','Training','PINN','Location','best');
        hold off;
        
        % Plot the training loss so far (log scale)
        subplot(1,2,2);
        semilogy(1:iter, lossHistory(1:iter), 'LineWidth',2);
        xlabel('Iteration'); ylabel('Loss');
        title('Training Loss History');
        drawnow;
    end
    
end

%% Helper Function: Model Loss
function [loss, gradients, lossData, lossPhys] = modelLoss( ...
    net, x_data, y_data, x_phys, mu, k)

    % Forward pass on data points (observations)
    y_pred_data = forward(net, x_data, 'Outputs','output');
    lossData = mean((y_pred_data - y_data).^2, 'all');
    
    % Forward pass on physics collocation points
    y_phys = forward(net, x_phys, 'Outputs','output');
    
    % 1) Sum the outputs to get a scalar for the derivative tape
    y_sum = sum(y_phys, 'all');
    
    % 2) First derivative: dy/dx
    dy_dx = dlgradient(y_sum, x_phys, 'EnableHigherDerivatives', true);
    
    % 3) Sum(dy_dx) => another scalar
    dy_dx_sum = sum(dy_dx, 'all');
    
    % 4) Second derivative: d2y/dx2
    d2y_dx2 = dlgradient(dy_dx_sum, x_phys, 'EnableHigherDerivatives', true);
    
    % ODE residual: y'' + mu*y' + k*y = 0
    physicsResidual = d2y_dx2 + mu * dy_dx + k * y_phys;
    lossPhys        = mean(physicsResidual.^2, 'all');
    
    % Total loss = data mismatch + small weight on the physics loss
    loss = lossData + 1e-4 * lossPhys;
    
    % Gradients wrt network learnable parameters
    gradients = dlgradient(loss, net.Learnables);
end
