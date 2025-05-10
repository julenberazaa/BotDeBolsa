clc
clear all
close all



Listado = dir('.\stocks');
for i=1:length(Listado)
    if length(strfind(Listado(i).name, '.csv'))>0
        FicheroConPath =strcat('Listado(i).name, .csv');
        datos = Irakurtzea(FicheroConPath);
        Cotizaciones=datos.open;
        Retornos=diff(Cotizaciones)
        RetornosMedios=[RetornosMedios, mean(Retornos)]
        VarianzaRetornos=[VarianzaRetornos, mean(Retornos)]
    end
    100*i/length(Listado)
end
save('ReaderBegining.mat', "VarianzaRetornos","RetornosMedios")