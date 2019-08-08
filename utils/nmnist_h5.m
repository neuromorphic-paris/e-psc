clc
clear 
close

%%% PARAMETERS %%%
first_saccade_only = true;
percentage_of_data = 100;

if exist('nmnist.h5', 'file') == 2
  delete('nmnist.h5');
end

train_dir = dir("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/N-MNIST/Train/**/*.bin");
test_dir = dir("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/N-MNIST/Test/**/*.bin");

if percentage_of_data < 100
    numberOfSamples = ceil((length(train_dir)*percentage_of_data)/100);
    train_dir = datasample(train_dir, numberOfSamples);
    
    numberOfSamples = ceil((length(test_dir)*percentage_of_data)/100);
    test_dir = datasample(test_dir, numberOfSamples);
end
                
for i = 1:length(train_dir)
    data = Read_Ndataset(strcat(train_dir(i).folder,'/',train_dir(i).name));
    if first_saccade_only
        data = temporalCrop(data, 0, 100000);
    end
    class_name = split(train_dir(i).folder,'/');
    label = repmat(str2double(class_name{end}),[size(data.x,1) 1]);
    h5create('nmnist.h5',strcat('/train/',train_dir(i).name),[5 size(data.x,1)]);
    h5write('nmnist.h5',strcat('/train/',train_dir(i).name),[data.ts';data.x';data.y';data.p';label']);
end

for i = 1:length(test_dir)
    data = Read_Ndataset(strcat(test_dir(i).folder,'/',test_dir(i).name));
    if first_saccade_only
        data = temporalCrop(data, 0, 100000);
    end
    class_name = split(test_dir(i).folder,'/');
    label = repmat(str2double(class_name{end}),[size(data.x,1) 1]);
    h5create('nmnist.h5',strcat('/test/',test_dir(i).name),[5 size(data.x,1)]);
    h5write('nmnist.h5',strcat('/test/',test_dir(i).name),[data.ts';data.x';data.y';data.p';label']);
end

% TD = Read_Ndataset(filename)
% returns the Temporal Difference (TD) events from binary file for the
% N-MNIST and N-Caltech101 datasets. See garrickorchard.com\datasets for
% more info
function TD = Read_Ndataset(filename)
    eventData = fopen(filename);
    evtStream = fread(eventData);
    fclose(eventData);

    TD.x    = evtStream(1:5:end); % pixel x address, with first pixel having index 1
    TD.y    = evtStream(2:5:end); % pixel y address, with first pixel having index 1
    TD.p    = bitshift(evtStream(3:5:end), -7); % polarity, 1 means off, 2 means on
    TD.ts   = bitshift(bitand(evtStream(3:5:end), 127), 16); % time in microseconds
    TD.ts   = TD.ts + bitshift(evtStream(4:5:end), 8);
    TD.ts   = TD.ts + evtStream(5:5:end);
end

% temporally crop a dataset
function [output] = temporalCrop(filename, ts1, ts2)
    % only keeps events that have timestamps within [ts1, ts2]
    mask = and(filename.ts >= ts1, filename.ts <= ts2);
    fields = fieldnames(filename);
    for i = 1:numel(fields)
        output.(fields{i}) = filename.(fields{i})(mask);
    end
end