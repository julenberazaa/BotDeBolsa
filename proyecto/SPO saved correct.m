clc
clear all
close all
%Parametros del algoritmo
nP=5; %Numero de particulas
nVar=2; %Numero de variables
nIteraciones=10; %Numero de iteraciones
w=0.1; %Inercia de las particulas
phi1Max = 0.1; %Parametro para modular la exploracion
phi2Max = 0.1; %Parametro para modular la exploracion

%iniciación de las particulas
x = random("uniform", -10,10,nVar,nP);
coste = zeros(1, nP);
for i=1:nP
    costes(i)=calcularCoste(x(:,i));%aqui le estamos diciendo 
    % que coja la columna nuero i de x
end
xOptimo = x;
costesOptimos = costes;
[costeOptimoGlobal, ref] = min(costesOptimos);
xOptimoGlobal = xOptimo(:, ref);
v=random("Uniforme", -0.1, 0.1, nVar, nP);


%Bucle principal
for t=1 : nIteraciones
    for i=1:nP
        phi1 = random("Uniform", 0, phi1Max,1,1);
        phi2 = random("Uniform", 0, phi2Max,1,1);
        v(:,i)= w*v(:,i)+phi1*(xOptimo(:,i)-x(:,i)) + phi2*(xOptimoGlobal-x(:,i));
        x(:,i)=x(:,i)+v(:,i);
        costes(i) = calcularCoste(x(:,i));
        if(costes(i) < costesOptimos(i))
           costesOptimos(i)=costes(i);
           xOptimo(:,i) = x(:,i);
           if(costeOptimoGlobal> costesOptimos(i))
              costeOptimoGlobal = costesOptimos(i);
           end
        end
    end
    %nos permite colocarnos en la primera grafica
    subplot(1,2,1);
    %vamos ha trazar donde se encuentra cada una de estas particulas
    plot(x(1,:),x(2,:), "xr");
    hold on;
    grid on;
    xlabel('x_1');
    ylabel('x_2');
    title('trayectoria de las particulas');

    suplot(1,2,2)
    plot(t, costeOptimoGlobal, ".b")
    hold on;
    grid on;
    xlabel('x_1');
    ylabel('x_2');
    title('trayectoria optima global');

end
%Función de coste
function [coste] = calcularCoste(x)
    coste = x(1)^2 + x(2)^2;
end