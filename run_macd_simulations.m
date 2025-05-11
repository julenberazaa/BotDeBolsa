% RUN_MACD_SIMULATIONS.M - Script para ejecutar todas las simulaciones de MACD
% Este script ejecuta todas las simulaciones relacionadas con la estrategia MACD

% Limpiar el espacio de trabajo
clc;
clear;
close all;

% Variable para rastrear errores
hasErrors = false;

% Mostrar información inicial
fprintf('=======================================================\n');
fprintf('       SIMULACIONES MACD PARA TRADING ALGORÍTMICO      \n');
fprintf('=======================================================\n\n');

% Asegurar que tenemos todas las rutas necesarias
try
    % Añadir rutas de src
    addpath(genpath('src'));
    % Añadir rutas de proyecto
    addpath('proyecto');
    % Añadir rutas de experimentos
    addpath('experiments');
    % Añadir rutas de datos
    addpath(genpath('data'));
    
    fprintf('✅ Rutas añadidas correctamente.\n\n');
catch err
    fprintf('❌ Error al añadir rutas: %s\n', err.message);
    return;
end

% Mostrar menú de opciones
fprintf('Seleccione una simulación para ejecutar:\n');
fprintf('1. Demo de MACD (serie sintética)\n');
fprintf('2. MACD vs Buy-and-Hold (entorno completo)\n');
fprintf('3. MACD vs RSI vs Buy-and-Hold (simulación directa)\n');
fprintf('4. MACD vs SPO vs Igual Peso (simulación directa)\n');
fprintf('5. Optimización de parámetros MACD\n');
fprintf('6. Comparación de todas las estrategias\n');
fprintf('7. MACD vs IA (simulación directa con red neuronal)\n');
fprintf('8. Ejecutar todas las simulaciones\n');
fprintf('0. Salir\n\n');

% Solicitar opción al usuario
option = input('Ingrese el número de la opción (0-8): ');

% Ejecutar la opción seleccionada
try
    switch option
        case 0
            fprintf('\nSaliendo...\n');
            return;
            
        case 1
            fprintf('\n=== Ejecutando Demo MACD ===\n');
            run demo_macd;
            fprintf('\n✅ Demo MACD completada con éxito.\n');
            
        case 2
            fprintf('\n=== Ejecutando MACD vs Buy-and-Hold ===\n');
            run run_experiment_macd;
            fprintf('\n✅ MACD vs Buy-and-Hold completada con éxito.\n');
            
        case 3
            fprintf('\n=== Ejecutando MACD vs RSI vs Buy-and-Hold ===\n');
            cd proyecto;
            run simulate_macd_vs_rsi;
            cd ..;
            fprintf('\n✅ MACD vs RSI vs Buy-and-Hold completada con éxito.\n');
            
        case 4
            fprintf('\n=== Ejecutando MACD vs SPO vs Igual Peso ===\n');
            cd proyecto;
            run simulate_macd_vs_others;
            cd ..;
            fprintf('\n✅ MACD vs SPO vs Igual Peso completada con éxito.\n');
            
        case 5
            fprintf('\n=== Ejecutando Optimización de Parámetros MACD ===\n');
            cd experiments;
            run optimize_macd_parameters;
            cd ..;
            fprintf('\n✅ Optimización de Parámetros MACD completada con éxito.\n');
            
        case 6
            fprintf('\n=== Ejecutando Comparación de Todas las Estrategias ===\n');
            cd experiments;
            run run_comparison_all;
            cd ..;
            fprintf('\n✅ Comparación de Todas las Estrategias completada con éxito.\n');
            
        case 7
            fprintf('\n=== Ejecutando MACD vs IA vs IA-MACD (Híbrido) ===\n');
            cd proyecto;
            run simulate_macd_vs_ia;
            cd ..;
            fprintf('\n✅ MACD vs IA completada con éxito.\n');
            
        case 8
            fprintf('\n=== Ejecutando Todas las Simulaciones ===\n');
            
            % Almacenar que estamos en modo "ejecutar todo"
            executeAllOption = true;
            
            fprintf('\n> Demo MACD\n');
            try
                run demo_macd;
            catch err
                fprintf('❌ Error en Demo MACD: %s\n', err.message);
                hasErrors = true;
            end
            
            fprintf('\n> MACD vs Buy-and-Hold\n');
            try
                run run_experiment_macd;
            catch err
                fprintf('❌ Error en MACD vs Buy-and-Hold: %s\n', err.message);
                hasErrors = true;
            end
            
            fprintf('\n> MACD vs RSI vs Buy-and-Hold\n');
            try
                cd proyecto;
                run simulate_macd_vs_rsi;
                cd ..;
            catch err
                fprintf('❌ Error en MACD vs RSI: %s\n', err.message);
                hasErrors = true;
                cd(fileparts(mfilename('fullpath'))); % volver al directorio principal
            end
            
            fprintf('\n> MACD vs SPO vs Igual Peso\n');
            try
                cd proyecto;
                run simulate_macd_vs_others;
                cd ..;
            catch err
                fprintf('❌ Error en MACD vs SPO: %s\n', err.message);
                hasErrors = true;
                cd(fileparts(mfilename('fullpath'))); % volver al directorio principal
            end
            
            fprintf('\n> MACD vs IA (Enfoque IA)\n');
            try
                cd proyecto;
                run simulate_macd_vs_ia;
                cd ..;
            catch err
                fprintf('❌ Error en MACD vs IA: %s\n', err.message);
                hasErrors = true;
                cd(fileparts(mfilename('fullpath'))); % volver al directorio principal
            end
            
            fprintf('\n> Optimización de Parámetros MACD\n');
            try
                cd experiments;
                run optimize_macd_parameters;
                cd ..;
            catch err
                fprintf('❌ Error en Optimización MACD: %s\n', err.message);
                hasErrors = true;
                cd(fileparts(mfilename('fullpath'))); % volver al directorio principal
            end
            
            fprintf('\n> Comparación de Todas las Estrategias\n');
            try
                cd experiments;
                run run_comparison_all;
                cd ..;
            catch err
                fprintf('❌ Error en Comparación de Estrategias: %s\n', err.message);
                hasErrors = true;
                cd(fileparts(mfilename('fullpath'))); % volver al directorio principal
            end
            
        otherwise
            fprintf('\n❌ Opción no válida. Por favor, seleccione un número entre 0 y 8.\n');
    end
catch err
    fprintf('\n❌ Error al ejecutar la simulación: %s\n', err.message);
    fprintf('En archivo: %s, línea %d\n', err.stack(1).file, err.stack(1).line);
    hasErrors = true;
    
    % Asegurar que volvemos al directorio principal
    try
        cd(fileparts(mfilename('fullpath')));
    catch
        % Ignorar error si ya estamos en el directorio principal
    end
end

% Resumen final para modos individuales (no ejecutar todo)
if ~exist('executeAllOption', 'var') || ~executeAllOption
    fprintf('\n=======================================================\n');
    fprintf('✅ Simulación completada.\n');
    fprintf('=======================================================\n');
else
    % Resumen final para modo ejecutar todo
    fprintf('\n=======================================================\n');
    if ~hasErrors
        fprintf('✅ Todas las simulaciones completadas con éxito.\n');
        fprintf('   Los resultados se han guardado en la carpeta "results".\n');
    else
        fprintf('⚠️ Simulaciones completadas con algunos errores.\n');
        fprintf('   Verifique los mensajes anteriores para más detalles.\n');
    end
    fprintf('=======================================================\n');
end 