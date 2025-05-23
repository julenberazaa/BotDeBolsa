function A = importarfichero(filename, dataLines)
%IMPORTFILE Import data from a text file
%  A = IMPORTFILE(FILENAME) reads data from text file FILENAME for the
%  default selection.  Returns the data as a table.
%
%  A = IMPORTFILE(FILE, DATALINES) reads data for the specified row
%  interval(s) of text file FILENAME. Specify DATALINES as a positive
%  scalar integer or a N-by-2 array of positive scalar integers for
%  dis-contiguous row intervals.
%
%  Example:
%  A = importfile("C:\Users\Julen Beraza\Desktop\4º\SISTEMAS APOYO A LA DECISIÓN\bd_stock price\stocks\A.csv", [1, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 06-Feb-2025 11:30:13

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [1, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 7);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"];
opts.VariableTypes = ["datetime", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "Date", "InputFormat", "yyyy-MM-dd", "DatetimeFormat", "preserveinput");

% Import the data
A = readtable(filename, opts);

end