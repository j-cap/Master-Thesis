%% Script for Defensio presentation

% date: 11.04.2021
% author: J. Weber

%% generate data
x = linspace(0,2*pi, 100)';
y = 1.5*sin(x) + x + randn(size(x))*0.25;

fig = figure();
scatter(x,y); xlabel("x"); ylabel("f(x)");
grid();

%% plot for linear model

X = [ones(size(x)) x];

clin = (X' * X) \ (X' * y);

fig = figure()
scatter(x,y); hold on;
plot(x, X * clin, 'LineWidth', 2');
grid()
title('Order l = 0')
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



