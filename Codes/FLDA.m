function[W, V, M] = FLDA(X)
    Images = [];
    for i = 1:size(X, 2)
        Images = [Images, X{i}];
    end
    
    [M, V] = ReconginitionVector(Images, 195);

    ProjectedImages = projectToPCA(Images, V, M);

    W = multiLDA(ProjectedImages);
end


function ldaVects = multiLDA(ProjectedImages)
    nClass = 65;
    nEachClass = 3;
    nFea = size(ProjectedImages, 1);

    mu = zeros(nFea, nClass);
    avg = mean(ProjectedImages, 2);

    for i = 1: nClass
        mu(:, i) = mean(ProjectedImages(:, ((i-1) * nEachClass + 1) : ((i-1) * nEachClass + nEachClass)), 2);
    end

    sw = zeros(nFea, nFea);
    for i = 1: nClass
        for j = 1: nEachClass
            x = ProjectedImages(:, (i - 1) * nEachClass + j);
            sw = sw + (x - mu(i)) * (x - mu(i))';
        end
    end

    sb = zeros(nFea, nFea);
    for i = 1 : nClass
        sb = sb + nEachClass * ((mu(i) - avg) * (mu(i) - avg)');
    end
    
    % find eigne values and eigen vectors of the (v)
    [ldaVects, ~]=eig(sb, sw);
end



% project to PCA subspace
function ProjectedImages = projectToPCA(FaceMat, Eigenfaces, avgImg)
    ProjectedImages = [];
    Train_Number = size(FaceMat, 2);
    for i = 1 : Train_Number
        pc = Eigenfaces' * (FaceMat(:, i) - avgImg);
        ProjectedImages = [ProjectedImages, pc]; 
    end
end


% Calculate "mean vector" ¡B "eigenvectors" and "eigenvalues"
function [avgImg, covVects] = ReconginitionVector(FaceMat, k)
    avgImg = mean(FaceMat, 2);
    diffTrain = FaceMat - avgImg;
    [eigVects, eigvals] = eig(diffTrain' * diffTrain);
    eigvals = diag(eigvals);
    eigvals_abs = abs(eigvals);
    [~, eigSortIndex] = sort(eigvals_abs, 'descend');
    eigSortIndex = eigSortIndex(1:k);
    covVects = diffTrain * eigVects(:, eigSortIndex);
    covVects = normalize(covVects);
end

