% MATLAB script to plot time series from a large CSV file

% Specify the file name
filename = 'B.csv'; % Change this to the name of your CSV file

% Read the data, skipping the first 11 rows
opts = detectImportOptions(filename, 'NumHeaderLines', 11);
data = readtable(filename, opts);

% Assuming the first column is the time, and the next three columns are the time series
time = data{:, 1}; % First column as time
series1 = data{:, 2}; % Second column as first time series
series2 = data{:, 3}; % Third column as second time series
series3 = data{:, 4}; % Fourth column as third time series

gain = 475000; % Taken from datasheet
responsivity = 0.6;
input_power = 0.018;    
cell_length = 0.012; % o 13 mm
delta_f = 1167680000;
delta_t = 0.00214916;
t_to_f_coeff = delta_f/delta_t;

frequency = time * t_to_f_coeff;

output_power =  series2 / (responsivity * gain);

absorption_coefficient = - log(output_power / input_power) / cell_length; 

absorption_coefficien_db = 10 * log10(absorption_coefficient);


reshaped_abs = reshape(absorption_coefficien_db, 20, []); % Reshape into 10 rows
mean_abs = mean(reshaped_abs, 1); % Mean along rows
reshaped_time = reshape(time, 20, []); % Reshape into 10 rows
mean_time = mean(reshaped_time, 1); % Mean along rows





% Plot the time series
figure;
%plot(time, series1, 'r', 'DisplayName', 'Series 1'); hold on;
%plot(frequency, absorption_coefficien_db, 'g', 'DisplayName', 'Series 2');
plot(mean_time, mean_abs, 'b', 'DisplayName', 'Series 3');
hold off;

% Add labels and legend
xlabel('Frequency');
ylabel('Value');
title(filename);
legend;
grid on;
