function wOptimoGlobal = obtenerSPO(RetornosMedios, VarianzaRetornos, alphaValue)

nP = 200;                           % Número de partículas
nVar = length(RetornosMedios);     % Número de activos
nIteraciones = 50;                 % Iteraciones del algoritmo
w = 0.1;                            % Inercia
phi1Max = 0.2;
phi2Max = 0.2;

% === Inicialización ===
costes = inf(1, nP);
x = zeros(nVar, nP);
v = random("Uniform", -0.1, 0.1, nVar, nP);

for i = 1:nP
    xcand = rand(nVar, 1);
    xcand = xcand / sum(xcand);
    coste = calcularCoste(xcand, alphaValue, RetornosMedios, VarianzaRetornos);
    x(:, i) = xcand;
    costes(i) = coste;
end

xOptimo = x;
costesOptimos = costes;
[~, ref] = min(costesOptimos);
xOptimoGlobal = xOptimo(:, ref);

% === Iteraciones principales ===
for t = 1:nIteraciones
    for i = 1:nP
        phi1 = rand * phi1Max;
        phi2 = rand * phi2Max;

        v(:, i) = w*v(:, i) + phi1*(xOptimo(:, i) - x(:, i)) + phi2*(xOptimoGlobal - x(:, i));
        x(:, i) = x(:, i) + v(:, i);

        x(:, i) = max(0, x(:, i));
        x(:, i) = x(:, i) / sum(x(:, i) + 1e-10);  % normalizar

        coste = calcularCoste(x(:, i), alphaValue, RetornosMedios, VarianzaRetornos);

        if coste < costesOptimos(i)
            costesOptimos(i) = coste;
            xOptimo(:, i) = x(:, i);
            if coste < calcularCoste(xOptimoGlobal, alphaValue, RetornosMedios, VarianzaRetornos)
                xOptimoGlobal = x(:, i);
            end
        end
    end
end

wOptimoGlobal = xOptimoGlobal;

end

% === Función de coste ===
function M = calcularCoste(w, alpha, rMedios, vRetornos)
    w = w(:);
    rMedios = rMedios(:);
    vRetornos = vRetornos(:);

    if ~(length(w) == length(rMedios) && length(w) == length(vRetornos))
        M = inf;
        return;
    end

    if any(w < 0) || sum(w) > 1.01
        M = inf;
        return;
    end

    M = sum((w.^2) .* vRetornos) - alpha * (w' * rMedios);
end
