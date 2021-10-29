clear; clc;
%% main
[trainingData, testingData] = prepareCellData();

[W, V, M] = FLDA(trainingData);

trainingData_pcaProject = projectToPCA(trainingData, V, M);
testingData_pcaProject = projectToPCA(testingData, V, M);

trainingData_ldaProject  = projectToLDA(trainingData_pcaProject, W);
testingData_ldaProject = projectToLDA(testingData_pcaProject, W);

recogRate = faceRecognition(trainingData_ldaProject, testingData_ldaProject);

fprintf('The overall recognition rate with FLDA is %0.3f%% \n', 100 * recogRate);


%% Functions
function recogRate=faceRecognition(trainingData_ldaProject, testingData_ldaProject)
    nClass = size(trainingData_ldaProject, 2);
    mu_LDA = cell(size(trainingData_ldaProject));
    for i = 1:nClass
        mu_LDA{i} = mean(trainingData_ldaProject{i}, 2);
    end

    matchCount = 0;
    nEachClass_test = size(testingData_ldaProject{1}, 2);
    for c = 1:nClass
        for i = 1:nEachClass_test
            classLabel = judgeFace(testingData_ldaProject{c}(:, i), mu_LDA);
            if classLabel == c
                matchCount = matchCount + 1;
            end
        end
    end
    recogRate = matchCount / (nClass * nEachClass_test);
end

function classLabel = judgeFace(test, mu_LDA)
    classNum = size(mu_LDA, 2);
    nEachClass_train = size(mu_LDA{1}, 2);
    minDist = inf;
    for c = 1:classNum
        for i = 1:nEachClass_train
            d = norm(test - mu_LDA{c}(:, i));
            if d < minDist
                minDist = d;
                classLabel = c;
            end
        end
    end
end


% project to LDA subspace
function ProjectedImages = projectToLDA(FaceMat, Eigenfaces)
    nClass = size(FaceMat, 2);
    nEachClass = size(FaceMat{1}, 2);
    ProjectedImages = cell(size(FaceMat));
    for i = 1:nClass
        pc = [];
        for j = 1:nEachClass
            pc = [pc, Eigenfaces' * FaceMat{i}(:, j)];
        end
        ProjectedImages{i} = pc;
    end
end

% project to PCA subspace
function ProjectedImages = projectToPCA(FaceMat, Eigenfaces, avgImg)
    nClass = size(FaceMat, 2);
    nEachClass = size(FaceMat{1}, 2);
    ProjectedImages = cell(size(FaceMat));
    for i = 1:nClass
        pc = [];
        for j = 1:nEachClass
            pc = [pc, Eigenfaces' * (FaceMat{i}(:, j) - avgImg)];
        end
        ProjectedImages{i} = pc;
    end  
end




