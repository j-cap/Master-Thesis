
clear all;
close all;
clc; 

% addpath('Funktionen')

% Intervallgrenzen
x_min	= -1;       % Minimum       
x_max	= 1;        % Maximum

% "Messpunkte" x
x	= x_min:0.025:x_max; 
n   = length(x);        % Anzahl an Punkten

% "Strukuroptimierung" --> Gitter
X	= x_min:0.4:x_max;  % Knotenpunkte
N_i = length(X);        % Anzahl an Knotenpunkten
N_f = N_i;              % Anzahl an Hut-Funktionen

% Zu approximierende Funktion:
% Polynom 3. Grades
a0	= 0.1;
a1	= 1;
a2	= 5;
a3	= 6;
f   = a0 + a1*x + a2*x.^2 + a3*x.^3;    % Funktion 
df  = a1 + a2*2*x + a3*3*x.^2;          % Ableitung der Funktion

% Aehnlicher Verlauf zu Waermeuebergangskoeffizient alpha
a0  = 2e4;
a1  = 5; 
f   = a0./( 1 + (a1*x).^2 );                % Funktion 
df  = -a0*2*a1^2*x./( 1 + (a1*x).^2 ).^2;   % Ableitung der Funktion

% Rauschen
noise	= 100*randn( 1, n );

% Funktion samt Rauschen
fn      = f + noise; 

%% Basisfunktionen: B-Splines

% B-Spline 1.Ordnung (linear, Hut-Funktion)
X_i = [ X(1), X, X(N_i) ];
b       = zeros( n, N_f );
db_dx   = zeros( n, N_f );
d2b_dx2 = zeros( n, N_f );
for i = 1:N_f
    for j = 1:n
        [ b(j,i), db_dx(j,i), d2b_dx2(j,i) ] = B_Spline_1( x(j), X_i(i), X_i(i+1), X_i(i+2) );
    end
end

% B-Spline 2.Ordnung (quadratisch)
X_i = [ X(1), X(1), X, X(N_i), X(N_i) ];

b       = zeros( n, N_f+1 );
db_dx   = zeros( n, N_f+1 );
d2b_dx2 = zeros( n, N_f+1 );
for i = 1:N_f+1
    for j = 1:n
        [ b(j,i), db_dx(j,i), d2b_dx2(j,i) ] = B_Spline_2( x(j), X_i(i), X_i(i+1), X_i(i+2), X_i(i+3) );
    end
end 
        
% Darstellen der Basisfunktion und dessen 1. und 2. Ableitung
figure('Name','Basisfunktionen');
pl0(1) = subplot(3,1,1);
hold on; grid on;
plot( x , b )
ylabel('b(x)');
xlabel('x');
title('Basisfunktionen')

pl0(2) = subplot(3,1,2);
hold on; grid on;
plot( x , db_dx )
ylabel('db/dx');
xlabel('x');
title('Erste Ableitung der Basisfunktionen')

pl0(3) = subplot(3,1,3);
hold on; grid on;
plot( x , d2b_dx2 )
ylabel('d^2b/dx^2');
xlabel('x');
title('Zweite Ableitung der Basisfunktionen')

linkaxes( pl0, 'x' );

%% Berechnen der Gewichte
gamma   = 0.0;

% Gewichte:
theta	=  b\( 1./( 1 + gamma )*fn )';

% Approximierte Funktion und erste Ableitung:
f_approx	= b*theta;          % Funktion 
df_approx    = db_dx*theta;     % Ableitung

% Darstellen der Ergebnisse
figure('Name','Approximation & Basisfunktionen');
pl1(1) = subplot(3,1,1);
hold on; grid on;
plot( x, fn, 'x' );
plot( x, f );
plot( x, f_approx, '--' );
legend('Orig','Noise','Approx')
ylabel('f(x)');
xlabel('x');
title('Funktionsapproximation')

pl1(2) = subplot(3,1,2);
hold on; grid on;
plot( x, df );
plot( x, df_approx );
ylabel('df/dx');
xlabel('x');
title('Steigung')

pl1(3) = subplot(3,1,3);
hold on; grid on;
plot( x, b );
ylabel('b(x)');
xlabel('x');
title('Basisfunktionen')

linkaxes( pl1, 'x' );







