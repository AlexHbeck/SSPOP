%% Sparse Sensor Placement Optimization for Prediction (SSPOP)
% by Alex C. Hollenbeck. See "Artificial Hair Sensor Placement Optimization
% on Airfoils for Angle of Attack Prediction," 2023 AIAA SciTech Paper.
% This code is derived from the SSPOC and SSPOR algorithms developed by
% Brunton, Manohar, Kutz, et. al. See especially DOI. 10.1137/15M1036713
% and  https://doi.org/10.48550/arXiv.1701.07569

clear all; close all;
warning('off','all');

%% definitions and data

% Full Model (161 nodes)
load XTR_full.mat % load velocity matrix for Regression
load XTE_full.mat % load velocity matrix for Testing

% Constrained model (129 nodes)
% load XTR_con.mat
% load XTE_con.mat

XTR = XTR_full; 
XTE = XTE_full;
% XTR = XTR_con;
% XTE = XTE_con;

load YTR.mat % load AoA matrix for regression
load YTE.mat % load AoA matrix for testing

Y = YTR;
Ytest = YTE;

% X = XTR_con';
X = XTR_full';

% G [1 x m] vector of AoA categories (ten of each in this case)
m   = length(Y); % there are 56 AoA categories in this data set
G   = [1:m]';
nx  = 1; % number of repeat measurements for each category = 1 here
ny  = length(X); % nodes in each measurement = 129 or 161
n   = nx * ny;    % total variables (candidate sensor locations)
m   = length(G);          % number of measurements
classes = unique(G);    % vector of classes
c   = numel(classes);     % number of classes 

%% VARIABLES
Q = 4; % number of sensors, chosen by user

% r and lambda may be chosen by experience or by exploring the whole 
%   space by loops. r varies from 1 to m while lambda may be zero,
%   1, 10, or 100. Sometimes r corresponds to the "elbow" in the 
%   singular value chart plotted next.
r       = 43; % truncate PSI to the first r modes or features
lambda  = 10; % The choice of lambda determines the sparsity in s.

%% Pre-process and plot with Singular Value Decomposition (a type of PCA)

[U, Sigma, V] = svd(X, 'econ'); % U-->Psi, sigma-->E, V-->V*  (Fig. 4)
dS = diag(Sigma)/sum(Sigma(:)); % the singular values of X
figure;
semilogy(dS(1:m), 'k');
xlabel('i');
ylabel('\sigma_i');
grid on;
title('Singular Value Spectrum');
set(gcf, 'Position', [100 100 300 300]);

%% Linear Discriminant Analysis (LDA)

d   = size(X, 1);
Psi = U(:, 1:r); % truncated basis PSI to first r modes
a   = Psi'*X; % a is the r-dimensional feature space of X
N   = zeros(1, c);   % mean measurement at each x in each class. 

% Find Centroid (only needed if we have repeat measurements).
for i = 1:c
    N(i) = sum(G==classes(i)); % all ones since one measurement per class
    centroid(:,i) = mean(X(:, G==classes(i)), 2);
end

Sw = zeros(d, d); % within-class variance (zero for singular classes)
    % simply each measurment in x minus the average measurement in x
for i = 1:c
    res = X(:,G==classes(i)) - centroid(:,i)*ones(1,N(i));
    Sw = Sw + (res)*(res)';
end

Sb = zeros(d, d); % between-class variance
for i = 1:c
    Sb = Sb + N(i) * (centroid(:,i)-mean(X,2))*(centroid(:,i)-mean(X,2))';
end

% solve for the eigenvalues of inv(Sw)*Sb,
% keep eigenvectors with c largest magnitude eigenvalues
[w, D] = eigs(pinv(Sw) * Sb, c);

% normalize w vectors (data is already normalized to freestream velocity)
for i = 1:size(w,2)
    w(:,i) = w(:,i) / sqrt(w(:,i)'*w(:,i));
end

w = w([1:r],:); % take first r rows of weights matrix
Xcls = w' * a;

%% Finding the Sparse Solution
% Step 1 - Find s
% solve Psi'*s = w, using the l1 norm to promote sparsity

epsilon = 1e-15; % error tolerance for reconstruction
unit = ones(c,1);

% must have CVX package: http://cvxr.com/cvx/download/ 
cvx_begin quiet
    variable s( n, c )
    minimize( norm(s(:),1) + lambda*norm(s*unit,1) ); 
    subject to
        norm(Psi'*s-w, 'fro') <= epsilon 
cvx_end

% Step 2 - Using sparse solution s, choose q sensor locations corresponding
% to the q rows of s containing at least one nonzero entry. In practice,
% a useful threshold is to find elements of s where |sij| >= ||s||F/(2*c*r)
S = abs(s(:,:)); 
for k = 1:ny % This creates a vector of the total size of each node in s
    S(k,:) = sort(abs(s(k,:)),'descend');  
    k = k+1;
end
Top = sort(S,'descend'); % this puts S in descending order. 
R = Top(Q);

sensors = S >= R;

[row,col,q] = find(sensors);
sensors = unique(row); % this is the Design Point (DP) sensor locations
q = numel(sensors); % should equal Q

% construct the measurement matrix Phi (ones correspond to sensors)
Phi = zeros(q, n);
for qi = 1:q,
    Phi(qi, sensors(qi)) = 1; % Phi is size [q x n]
end;

%% Regression and testing of DP
DP = sensors(:)' 
out = zeros(size(X)); 
out(DP,:) = X(DP,:); 
out = out';

Bdp = regress(Y,out); % this regression gives a rank deficient warning
              % but this is ok becuase the columns are mostly zeros
              % therefore warnings are suppressed.
pred = XTE*Bdp;
RMSE = sqrt(mean((pred-Ytest).^2));

fprintf('Predictive accuracy by RMSE of %i SSPOC sensors is %g degrees \n',...
    q, RMSE)


