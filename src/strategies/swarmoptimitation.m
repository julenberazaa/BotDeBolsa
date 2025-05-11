clc;
clear all;
close all;
load ('ReaderBegining.mat');
%mas adelante hacer un scrip que lanza este problama con diferentes
%PROS:no necesitamos derivadas o expresión analitica, no hay limitaciones
%en la funcion de coste(sin discontinuidad), los minimos locales los evita
%facilmente, siempre a mejor
%CONTRAS: no podemos asegurar haber obtenido el optimo global, ++nVar el
%espacio de busqueda crece de manera exponencial(algoritmo voraz), ajustar
%manualmente los parametros no es evidente
%valores%
%parametros del algoritmo%
nP=20;%numero particulas
nVar=length(RetornosMedios);%numero variables
nItraciones=50;%numero iteraciones
w=0.1;%innercia particulas
phi1Max=0.25;%Parametro para modular la exploracion
phi2Max=0.25;%Parametro para modular la explotacion

%Inicializacion de las particulas%
x = zeros(nVar,nP);
costes= zeros(1,nP);
alpha=0.1;
for i=1:nP
    costes(i)= inf;
    %solo esten en el primer cuadrante, particulas no sean negativas
    while costes(i)==inf
    xcand= random("Uniform",0,1,nVar,1);
    xcand= xcand/sum(xcand);
    costes(i) = calcularCoste(xcand,alpha, RetornosMedios,VarianzaRetornos);
    end
    x(:,i) = xcand;
end 
xOptimos =x;
costesOptimos= costes;
[costeOptimoGlobal,ref]=min(costesOptimos); % minimo de costesOptimos, ref es que posicion se encuentra
xOptimoGlobal=xOptimos(:,ref); % guardando la posicion que corresponde con ese valor
v= random("uniform",-0.1,0.1,nVar,nP);

%Bulcle principal%
for t=1:nItraciones
    for i=1:nP
        phi1=random("uniform",0,phi1Max,1,1);% coheficiente exploracion
        phi2=random("uniform",0,phi2Max,1,1);% coheficiente explotacion
        v(:,i) = w*v(:,i)+phi1*(xOptimos(:,i)-x(:,i))+phi2*(xOptimoGlobal-x(:,i)); %velocidad funcion
        x(:,i)= x(:,i)+v(:,i);%calculo de la particula de posicion
        costes(i)= calcularCoste(x(:,i),alpha, RetornosMedios,VarianzaRetornos); % calcular costes de la particula de posicion i
        if costes(i)<costesOptimos(i)
            costesOptimos(i)=costes(i);
            xOptimos(:,i)=x(:,i);
            xOptimoGlobal = xOptimos(:,i); % 🔹 Asegura que guardamos la mejor posición global
            if costesOptimos(i)< costeOptimoGlobal
                costeOptimoGlobal = costesOptimos(i);
            end
        end
    end
    subplot(1,2,1);
    plot(x(1,:),x(2,:),'xr');
    hold on;
    grid on;
    xlabel("x_1");
    ylabel("x_2");
    title("Trayectoria de las particulas");
    subplot(1,2,2);
    plot(t,costeOptimoGlobal,'.b');
    hold on;
    grid on;
    xlabel("Numero iteraciones");
    ylabel("Trayectoria del CosteOptimoGlobal");
   
end
%funcion de coste, sería markowiz%
% function [coste] = calcularCoste (x);
%     coste = x(1)^2+x(2)^2;
%     %lamda= 10;
%     %coste= coste+ lamda*exp((x(1)+x(2)-10)^2);
%     %if( x(1)<0 || x(2)<0 ||x(1)+x(2)>10)
%     %   coste = inf;
%     %end
%     h=0.5;
%     if x(1)+ x(2)>h || x(1)+x(2)<h
%         coste = inf;
%     end
% end
%MARKOWITZ calcular costes
function [M]= calcularCoste(w,alpha, RetornosMedios,VarianzaRetornos)
    %Covarianza = diag(VarianzaRetornos);
    %M = w*Covarianza*w'-alpha*w*RetornosMedios';
    M = sum((w'.^2).*VarianzaRetornos)-alpha*w'*RetornosMedios';
    % se evalian las restricciones
    RefAptas = w;
    RefAptas(RefAptas>1)=[];
    RefAptas(RefAptas<0)=[];
    if length(RefAptas)<length(w)
        M=inf;
    end
    if sum(w)>1
        M=inf;
    end
end
%actualizacion posiciones%

%actualizacion velocidad%

%posicion optima%

