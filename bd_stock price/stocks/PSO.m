clc
clear all
close all
load('Resultados.mat')
%Parámetros del algoritmo
nP = 50; %Número de partículas
nVar = length(RetornosMedios); %Número de variables
nIreraciones = 1000;
w = 0.3; %Inercia de las partículas
phi1Max = 0.2; %Parámetro para modular la exploración
phi2Max = 0.2; %Parámetro para modular la explotación


%Inicialización de las partículas
x = zeros(nVar, nP);
costes = zeros(1,nP);
for i = 1 : nP
    costes(i) = inf;
    while costes(i) == inf
        xcand = random("uniform", 0, 10, nVar, 1);
        costes(i) = CalcularCoste(xcand);
    end
    x(:,i) = xcand;
end
xOptimos = x;
costesOptimos = costes;
[costeOptimoGlobal, ref] = min(costesOptimos) % está calculando el valor más pequeño de CostesOptimos y la mete en la variable CosteOptimoGlobal
xOptimoGlobal = xOptimos(:,ref); %ref devuelve el valor mínimo y en que posición
v = random("Uniform", -0.1, 0.1, nVar, nP); %Inicialización de la velocidad

%Bucle principal
for t = 1 : nIreraciones
    for i = 1 : nP
        phi1 = random("Uniform", 0, phi1Max, 1, 1); %El 1, 1 es una fila una columna
        phi2 = random("Uniform", 0, phi2Max, 1, 1);
        v(:,i) = w*v(i) + phi1*(xOptimos(:,i) - x(:,i)) + phi2*(xOptimoGlobal - x(:,i)); %Cálculo de velocidad
        x(:, i) = x(:, i) + v(:,i);
        costes(i) = CalcularCoste(x(:,i));
        if costes(i) < costesOptimos(i)
            costesOptimos(i) = costes(i);
            xOptimos(:,i) = x(:,i);
            if costeOptimoGlobal > costesOptimos(i)
                costeOptimoGlobal = costesOptimos(i)
                xOptimoGlobal = xOptimos (:,i);
            end
        end
    end
        subplot(1, 2, 1)
        plot(x(1, :), x(2, :), 'xr');
        hold on;
        grid on;
        xlabel('x_1');
        ylabel('x_2');
        title('Trayectoria de las particulas')
        subplot(1, 2, 2);
        plot(t, costeOptimoGlobal, '.b');
        hold on;
        grid on;
        xlabel('Iteraciones');
        ylabel('Costes');
        title('Trayectoria del coste optim global')
end
subplot(1, 2, 1);
x1vec=[0;10];
x2vec=10-x1vec;
plot(x1vec, x2vec, 'k');
plot(xOptimoGlobal(1), xOptimoGlobal(2), 'og');
%Función de coste
function [coste] = CalcularCoste(x)
    coste = x(1)^2 + x(2)^2;
    lambda = 100;
    %coste = coste + lambda * exp((x(1)+x(2)-10)^2);
    %if x(1) < 0 || x(2) < 0 || x(1)+x(2)>10
     %   coste = inf;
    %end
    h=0.05;
    if x(1)+x(2)-10>h || x(1)+x(2)-10<-h
        coste=inf;
    end
end