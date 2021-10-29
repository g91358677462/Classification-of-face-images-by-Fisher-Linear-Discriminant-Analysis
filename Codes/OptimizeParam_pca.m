% = = = = = = = = = = = = = = = = = = = = %
% Student, ID: 107522121, Name: ¯Î¥¿­è  %
% = = = = = = = = = = = = = = = = = = = = %
clear; clc;
%% main
[trainingData, testingData] = loadTrainTestSet();
bestRecogRate = 0;
bestK = 0;
for k = 1:195
    recogRate = recog_pca(k);
    if recogRate > bestRecogRate
        bestRecogRate = recogRate;
        bestK = k;
    end
end

fprintf('When k = %d, Best recognition rate = %0.3f%% \n', bestK,100 * bestRecogRate);

%% Functions
% Reconginition rate of pca
function recogRate = recog_pca(k)
    classNum = 65;
    trainNum = 3;
    testNum = 18;
    [trainingData, testingData] = loadTrainTestSet();
    [m, V, D] = ReconginitionVector(trainingData, k);

    % Get project coefficients of training data
    pcTrainSet = {};
    for c = 1:classNum
        tmpPC = {};
        s = (c-1) * trainNum + 1;
        e = c * trainNum;
        for i = s:e
            pc = V' * (trainingData(:, i) - m);
            tmpPC = {tmpPC{:}, pc};
        end
        pcTrainSet = {pcTrainSet{:}, tmpPC};
    end

    % Get project coefficients of testing data
    pcTestSet = {};
    for c = 1:classNum
        tmpPC = {};
        s = (c-1) * testNum + 1;
        e = c * testNum;
        for i = s:e
            pc = V' * (testingData(:, i) - m);
            tmpPC = {tmpPC{:}, pc};
        end
        pcTestSet = {pcTestSet{:}, tmpPC};
    end

    % Face recognition
    matchCount = 0;
    for c = 1:classNum
        for i = 1:testNum
            classLabel = judgeFace(pcTestSet{c}{i}, pcTrainSet);
            if classLabel == c
                matchCount = matchCount + 1;
            end
        end
    end
    
    recogRate = matchCount / (classNum * testNum);
    
    % Print result
    fprintf('Recognition rate = %0.3f%% \n', 100 * recogRate);
end

% Identification picture
function classLabel = judgeFace(pc_test, pcTrainSet)
    classNum = 65;
    trainNum = 3;
    minDist = inf;
    for c = 1:classNum
        for i = 1:trainNum
            d = norm(pc_test - pcTrainSet{c}{i});
            if d < minDist
                minDist = d;
                classLabel = c;
            end
        end
    end
end

% Load training and testing data
function [trainingData, testingData] = loadTrainTestSet()
    trainingData = [];
    testingData = [];
    trainIndexSet = [7, 10, 19];
    for pid = 1:65
        dataDir = ['../data/', num2str(pid), '/'];
        for j = 1:21
            if any(j == trainIndexSet)
                d = double(imread([dataDir, sprintf('%d.bmp', j)]));
                trainingData = [trainingData, d(:)];
            else
                d = double(imread([dataDir, sprintf('%d.bmp', j)]));
                testingData = [testingData, d(:)];
            end
        end
    end
end

% Calculate "mean vector" ¡B "eigenvectors" and "eigenvalues"
function [avgImg, covVects, eigvals] = ReconginitionVector(FaceMat, k)
    avgImg = mean(FaceMat, 2);
    diffTrain = FaceMat - avgImg;
    [eigVects, eigvals] = eig(diffTrain' * diffTrain);
    eigvals = diag(eigvals);
    eigvals_abs = abs(eigvals);
    [~, eigSortIndex] = sort(eigvals_abs, 'descend');
    eigSortIndex = eigSortIndex(1:k);
    eigvals = eigvals(eigSortIndex);
    covVects = diffTrain * eigVects(:, eigSortIndex);
    covVects = normalize(covVects);
end

% Normalize vectors
function v = normalize(v)
    [~, numV] = size(v);
    for i = 1:numV
        v(:, i) = v(:, i) / norm(v(:,  i));
    end
end