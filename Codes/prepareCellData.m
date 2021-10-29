% Load training and testing data
function [trainingData, testingData] = prepareCellData()
    trainingData = {};
    testingData = {};
    trainIndexSet = [7, 10, 19];
    for pid = 1:65
        dataDir = ['../data/', num2str(pid), '/'];
        tmp_trainingData = [];
        tmp_testingData = [];
        for j = 1:21
            if any(j == trainIndexSet)
                d = double(imread([dataDir, sprintf('%d.bmp', j)]));
                tmp_trainingData = [tmp_trainingData, d(:)];
            else
                d = double(imread([dataDir, sprintf('%d.bmp', j)]));
                tmp_testingData = [tmp_testingData, d(:)];
            end
        end
        trainingData = {trainingData{:}, tmp_trainingData};
        testingData = {testingData{:}, tmp_testingData};
    end
end
