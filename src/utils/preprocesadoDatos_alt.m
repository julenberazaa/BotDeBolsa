clc;
close all;
RetornosMedios =[];
VarianzaRetornos =[];
Listado = dir(".\stocks\");
Inversiones.nombres=[];
for i=1:length(Listado)
    if length(strfind(Listado(i).name,".csv")) >0
        Inversiones(i).nombres=Listado(i).name;
        FicheroConPath = strcat(".\stocks\",Listado(i).name);
        datos = importarfichero(FicheroConPath);
        Cotizaciones = datos.Open;
        Retornos = diff(Cotizaciones);
        RetornosMedios =[RetornosMedios,mean(Retornos)];
        VarianzaRetornos =[VarianzaRetornos, var(Retornos)];
    end
    100*i/length(Listado)
end
save('Resultados.mat', "VarianzaRetornos",...
    "RetornosMedios", "Inversiones");