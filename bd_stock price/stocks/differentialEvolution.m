% Parámetros iniciales del algoritmo Differential Evolution
numIteraciones = 5;
numAgentes = 100;
cr = 0.5;
F = 1;
numVariables = 2;

% Inicialización de los agentes
x = random("Uniform", -10, 10, numVariables, numAgentes);
costes = zeros(1, numAgentes);
for k = 1:numAgentes
    costes(k) = calcularCoste(x(:, k));
end

% Bucle principal del algoritmo de optimización
for t = 1:numIteraciones
    for i = 1:numAgentes
        % Escoger Xa, Xb y Xc usando índices separados
        refVec = 1:numAgentes;
        
        % Seleccionar Xa
        indiceA = random('unid', numAgentes, 1, 1);
        Xa = x(:, refVec(indiceA));
        refVec(indiceA) = [];
        
        % Seleccionar Xb
        indiceB = random('unid', numAgentes-1, 1, 1);
        Xb = x(:, refVec(indiceB));
        refVec(indiceB) = [];
        
        % Seleccionar Xc
        indiceC = random('unid', numAgentes-2, 1, 1);
        Xc = x(:, refVec(indiceC));
        
        % Crear el vector Y = Xc + F*(Xa - Xb)
        Y = Xc + F * (Xa - Xb);
        
        % Cruce de Y y Xi (agente actual) con probabilidad cr
        T = x(:, i);
        for j = 1:numVariables
            if random('Uniform', 0, 1, 1, 1) < cr
                T(j) = Y(j);
            end
        end
        
        % Evaluar el agente T y seleccionar el mejor entre T y Xi
        ct = calcularCoste(T);
        if ct < costes(i)
            costes(i) = ct;
            x(:, i) = T;
        end
    end
    
    % Graficar la evolución de los agentes y del coste mínimo
    subplot(1, 2, 1)
    plot(x(1, :), x(2, :), 'xr');
    grid on
    xlabel('x_1');
    ylabel('x_2');
    title('Evolución de los agentes')
    hold on;
    
    subplot(1, 2, 2)
    plot(t, min(costes), '.b');
    grid on
    xlabel('Iteraciones');
    ylabel('Coste mínimo');
    title('Evolución del coste mínimo')
    hold on;
    pause(1)
end

% Función de coste
function [coste] = calcularCoste(xAgente)
    coste = xAgente(1)^2 + xAgente(2)^2;
end
