%% Script for Defensio presentation

% date: 11.04.2021
% author: J. Weber

%% generate data
x = linspace(0,2*pi, 100)';
y = 1.5*sin(x) + x + randn(size(x))*0.25;


%% plot for linear model (and polynomial)

X = [ones(size(x)) x x.^2 x.^3];

Xlin = X(:,1:2);
Xquad = X(:,1:3);
Xcub = X;
c_lin = (Xlin' * Xlin) \ (Xlin' * y);
c_quad = (Xquad' * Xquad) \ (Xquad' * y);
c_cub = (Xcub' * Xcub) \ (Xcub' * y);

fig = figure()
scatter(x,y); hold on;
plot(x, Xlin* c_lin, 'LineWidth', 2');
plot(x, Xquad * c_quad, 'LineWidth', 2');
plot(x, Xcub * c_cub, 'LineWidth', 2');
grid(); legend('Data', 'Linear', 'Quadratic', 'Cubic');
%title('Order l = 0')
xlabel("x"); ylabel("f(x)");
legend("Data", "Linear Model", "Quad. Model", "Cubic Model");
ax = gca;
ax.FontSize = 15; 

%% generate B-spline of order 0, 1, 2, 3
nr_splines = 25;
sorder = 0;

B0 = Bspline.basismatrix(x, nr_splines, 0, "e");
c0 = Bspline.fit(x, y, nr_splines, 0, "e");

B1 = Bspline.basismatrix(x, nr_splines, 1, "e");
c1 = Bspline.fit(x, y, nr_splines, 1, "e");

B2 = Bspline.basismatrix(x, nr_splines, 2, "e");
c2 = Bspline.fit(x, y, nr_splines, 2, "e");

B3 = Bspline.basismatrix(x, nr_splines, 3, "e");
c3 = Bspline.fit(x, y, nr_splines, 3, "e");

%% plot B-spline of order 0, 1, 2, 3

ax11 = subplot(2,2,1);
scatter(x,y); hold on;
plot(x, B0*c0, 'LineWidth', 2);
plot(x, B0)
grid; xlim([min(x), max(x)]);
title('Order l = 0')
ax = gca;
ax.FontSize = 15;

ax12 = subplot(2,2,2);
scatter(x,y); hold on;
plot(x, B1*c1, 'LineWidth', 2);
plot(x, B1)
grid; xlim([min(x), max(x)]);
title('Order l = 1')
ax = gca;
ax.FontSize = 15;


ax21 = subplot(2,2,3);
scatter(x,y); hold on;
plot(x, B2*c2, 'LineWidth', 2);
plot(x, B2); grid; xlim([min(x), max(x)]);
title('Order l = 2')
ax = gca;
ax.FontSize = 15;


ax22 = subplot(2,2,4);
scatter(x,y); hold on;
plot(x, B3*c3, 'LineWidth', 2);
plot(x, B3); grid; xlim([min(x), max(x)]);
title('Order l = 3')
ax = gca;
ax.FontSize = 15;

%% Plot the basis matrices
ax11 = subplot(2,2,1);
imagesc(B0') %,'CDataMapping','scaled');
title('Order l = 0')
ax = gca;
ax.FontSize = 15;
colorbar()

ax12 = subplot(2,2,2);
imagesc(B1','CDataMapping','scaled');
title('Order l = 1')
ax = gca; colorbar()
ax.FontSize = 15;


ax21 = subplot(2,2,3);
imagesc(B2','CDataMapping','scaled');
title('Order l = 2')
ax = gca;
ax.FontSize = 15; 
colorbar(); caxis([0,1]);


ax22 = subplot(2,2,4);
imagesc(B3','CDataMapping','scaled');
title('Order l = 3')
ax = gca;
ax.FontSize = 15;
colorbar(); caxis([0,1]);

%% Plot P-spline fits for different \lambdas 
nr_splines = 50;

B3 = Bspline.basismatrix(x, nr_splines, 3, "e");

c3p_0 = Bspline.fit_Pspline(x,y,0,nr_splines, 3, "e");
c3p_1 = Bspline.fit_Pspline(x,y,1,nr_splines, 3, "e");
c3p_10 = Bspline.fit_Pspline(x,y,10,nr_splines, 3, "e");
c3p_72 = Bspline.fit_Pspline(x,y,72,nr_splines, 3, "e");
c3p_1000 = Bspline.fit_Pspline(x,y,5000,nr_splines, 3, "e");

fig = figure();
scatter(x,y); hold on;
plot(x, B3*c3p_1);
plot(x, B3*c3p_10);
plot(x, B3*c3p_72);
plot(x, B3*c3p_1000);

%%
ax11 = subplot(2,2,1);
scatter(x,y); hold on;
plot(x, B3*c3p_0, 'LineWidth', 2);
grid; xlim([min(x), max(x)]);
title('$$\lambda = 0$$', 'interpreter', 'latex')
ax = gca;
ax.FontSize = 15;

ax12 = subplot(2,2,2);
scatter(x,y); hold on;
plot(x, B3*c3p_1, 'LineWidth', 2);
grid; xlim([min(x), max(x)]);
title('$$\lambda = 1$$', 'interpreter', 'latex')
ax = gca;
ax.FontSize = 15;

ax21 = subplot(2,2,3);
scatter(x,y); hold on;
plot(x, B3*c3p_72, 'LineWidth', 2);
grid; xlim([min(x), max(x)]);
title('$$\lambda = 72$$', 'interpreter', 'latex')
ax = gca;
ax.FontSize = 15;

ax22 = subplot(2,2,4);
scatter(x,y); hold on;
plot(x, B3*c3p_1000, 'LineWidth', 2);
grid; xlim([min(x), max(x)]);
title('$$\lambda = 1000$$', 'interpreter', 'latex')
ax = gca;
ax.FontSize = 15;


%% Increasing behavior

dUncon = {["s(1)", 25, "none", 0, "e"]};
[cUnc, Bunc] = Stareg.fit(dUncon, x, y);

dcon = {["s(1)", 25, "inc", 3000, "e"]};
[c, B] = Stareg.fit(dcon, x, y);

fig = figure(); hold on;
scatter(x, y);
plot(x, B*c, "LineWidth", 2); plot(x, Bunc*cUnc, "LineWidth", 2); 
grid; xlim([min(x), max(x)]);
legend("Data", "SC-P-spline", "P-spline");
title('$$Inc. \ Constraint, \lambda_c=3000$$, d=25, l=3', 'interpreter', 'latex')
ax = gca;
ax.FontSize = 15;
saveas(fig, "defensio/SCPspline_increasing", "png");







