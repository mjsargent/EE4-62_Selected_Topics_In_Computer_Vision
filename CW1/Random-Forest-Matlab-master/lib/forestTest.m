function [Yhard, Ysoft, allLeafIdxs] = forestTest(model, X, opts)
    % X is NxD, where rows are data points
    % model comes from forestTrain()
    % Yhard are hard assignments to X's, Ysoft is NxK array of
    % probabilities, where there are K classes.
    
    if nargin<3, opts= struct; end
    
    numTrees= length(model.treeModels);
    allLeafIdxs = cell(numTrees,1);
    %u= model.treeModels{1}.classes; % Assume we have at least one tree!
    u = [1:10]'; % CHANGED
    Ysoft= zeros(size(X,1), 10); %length(u) CHANGED
    for i=1:numTrees
        [~, ysoft, allLeafIdxs{i}] = treeTest(model.treeModels{i}, X, opts);
        Ysoft= Ysoft + ysoft;
    end
    
    Ysoft = Ysoft/numTrees;
    [~, ix]= max(Ysoft, [], 2);
    Yhard = u(ix);
end
