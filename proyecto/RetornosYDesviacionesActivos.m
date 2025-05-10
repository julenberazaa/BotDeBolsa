% Cargar datos
load('ReaderBeginingDLR.mat');  % 'RetornosMedios' y 'VarianzaRetornos'

% Cálculos
numActivos = size(RetornosMedios, 1);
numTiempos = size(RetornosMedios, 2);
retornosPromedio = mean(RetornosMedios, 2);
desviaciones = sqrt(VarianzaRetornos);

% Crear una figura
figure;

for i = 1:numActivos
    subplot(ceil(numActivos/2), 2, i);  % Organizar en subplots (2 columnas)
    plot(1:numTiempos, RetornosMedios(i,:), 'b'); hold on;
    
    % Líneas de retorno medio y desviaciones
    yline(retornosPromedio(i), 'r-', 'Media');
    yline(retornosPromedio(i) + desviaciones(i), 'g--', '+1 Desv Est');
    yline(retornosPromedio(i) - desviaciones(i), 'g--', '-1 Desv Est');
    
    grid on;
    title(sprintf('Activo %d', i));
    xlabel('Tiempo');
    ylabel('Retorno');
end

sgtitle('Retornos históricos por activo con media y desviación estándar');
