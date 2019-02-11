function [imageHist] = findImageHistogram(model, descriptors_in, opts)
    [~, ~, allLeafIdxs] = forestTest(model, descriptors_in, opts);
    d = opts.depth;
    nd = 2^d - 1;
    imageHist = zeros(opts.numTrees*2^(d-1),1);
    treeHist = zeros(2^(d-1),1);
    for tree = 1:opts.numTrees
        histIdx = 1;
        for leafIdx = (nd+1)/2 : nd
            idxSum = sum(allLeafIdxs{tree}(:, leafIdx));
            treeHist(histIdx,1) = idxSum;
            histIdx = histIdx + 1;
        end
        imageHist(1+(tree-1)*2^(d-1):tree*2^(d-1),1) = treeHist;%/sum(treeHist);
    end
    imageHist = imageHist'/sum(imageHist);
end