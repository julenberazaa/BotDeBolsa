clc
clear all
close all



Listado = dir('.\stocks');
RetornosMedios=[];
VarianzaRetornos=[];
Inversiones.nombres=[];
for i=1:length(Listado)
    if length(strfind(Listado(i).name, '.csv'))>0
        Inversiones(i).nombres=Listado(i).name
        FicheroConPath =strcat('.\stocks\', Listado(i).name);
        datos = Irakurtzea(FicheroConPath);
        Cotizaciones=datos.Open;
        Retornos=diff(Cotizaciones);
        RetornosMedios=[RetornosMedios, mean(Retornos)];
        VarianzaRetornos=[VarianzaRetornos, mean(Retornos)];
    end
    100*i/length(Listado)
end
save('ReaderBegining.mat', "VarianzaRetornos", ...
    "RetornosMedios", "Inversiones")
