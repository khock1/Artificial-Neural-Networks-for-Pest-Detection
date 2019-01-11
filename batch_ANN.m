% Author: Karlo Hock, University of Queensland. (c) 

% Run an ensebmle of Artificial Neural Networks (ANNs) for detection of pests in habitat patches
% Set up ANN
% Train each ANN separately using the training set
% Retain the ANN that performs the best 
% Test the performance of the retained ANN on the new test dataset
% Plot the confusion matrix to see how well the retained ANN is handling the new data

clear;
% load dummy datasets; file with the data can be dowloaded from this same folder;
% the parameters and their values are based on exploratory analysis of coral reefs with crown-of-thorns starfish outbreaks
% the datasets contain 8 'environmental' variables, which can be continuous, discrete or binary
% the outputs are binary, 1 = presence 0 = absence
load('annbatchtrain.mat')
input = train_data';% desginate data part of the ANN analysis
target = train_output';% designate output part of the ANN analysis
trainFcn = 'trainscg'; % training function
hiddenLayerSize = 20; % hidden layer
net = patternnet(hiddenLayerSize, trainFcn);% set up ANN
net.input.processFcns = {'removeconstantrows', 'mapminmax'};
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;% 70% of the training data will be used to train ANN
net.divideParam.valRatio = 15/100;% 15% of the training data will be used to validate ANN
net.divideParam.testRatio = 15/100;% 15% of the training data will be used to test ANN
net.performFcn = 'crossentropy';  % Cross-Entropy
net.trainParam.showWindow = false;% no visualisation

% criteria for determining which the best performign ANN to retain
reject_all = 0.5;% overall rejection rate, overall fraction of misclassified samples must be lower than this value
reject_tp = 0.5;% we also need to have a true positive rate that is higher than this value...
reject_tn = 0.3;% ...and a true negative rate that is higher than this value, which should generally be less than true postiive criterion because we do not want to miss too many infested patches ...
reject_dr = 0.7;% ... and finally we also need to have an overall precision that is higher than this value to make sure that we are not missing too many patches with pests
stored_net = net;% container for the best performing network

% run an ensemble of 200 individually trained ANNs to get the best performing one;
% since each ANN will be slightly different based on how traning dataset was
% randomly partitioned etc., we want to run a bunch of them to get the one with the
% best perfomance to test on the new data
for ann = 1:200
    [net,tr] = train(net, input, target);% train a network with the training data usign the ANN specification above
    output = net(input);% get the output of the ANN as probabilties of classification
    [misclass_samples, conf_mat] = confusion(train_output', output);% get the fraction of misclassified samples and a confusion matrix for the traning dataset
    truepos = conf_mat(2, 2)/(sum(conf_mat(:, 2)));% calculate the true postive rate of this ANN
    trueneg = conf_mat(1, 1)/(sum(conf_mat(:, 1)));% calculate the true negative rate of this ANN
    detectrate = conf_mat(2, 2)/(sum(conf_mat(2, :)));% calculate precision/positive predictive value of this ANN
    if ((truepos >= reject_tp) + (trueneg >= reject_tn) + (detectrate >= reject_dr) + (misclass_samples <= reject_all)) == 4% iff all the criteria are satisified
        % use the classification performance values from this ANN as the new set of criteria for future ANN rejection
        reject_all = misclass_samples;
        reject_tp = truepos;
        reject_tn = trueneg;
        reject_dr = detectrate;
        % store the new best performing ANN and its output
        stored_net = net;
        stored_output = output;
    end
end

% now test the best peforming ANN on some new data that was not included in the traning set
ANN_output = stored_net(new_data');% get the output using the stored ANN
[Lc,Lcm] = confusion(new_output', ANN_output);% get the confusion matrix to compare ANN predictions with the actual output for the new dataset
plotconfusion(new_output', ANN_output)% plot this confusion matrix to see how many true negatives etc. we would have in the new dataset with the retained ANN
