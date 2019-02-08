clear all; close all; rng(1)
%% 0. Initialisation
init; clc;
%% 1. Generate data
vocab_size = 200;
[data_train, data_test] = getData(vocab_size);
%% 2. Random forest training
% choose tree options
number_runs = 10;
depth = [2,5,8];
numTree = [10,100,1000,10000];
numSplits = [20,50,100];
classifierId = [1,2,5];
classifierCommitFirst = false;
bagSizes = [50,100,200];
decChoice = [1]; % / 1 = entropy, 2 = Gini index
% standardise data to zero mean, unit variance  === DON'T DO THIS ===
%data_train = bsxfun(@rdivide, bsxfun(@minus, data_train, mean(data_train)), var(data_train) + 1e-10);
%data_test = bsxfun(@rdivide, bsxfun(@minus, data_test, mean(data_train)), var(data_train) + 1e-10);
true_class_vector = reshape((ones(10,15).*[1:10]')',[150,1]);
%%
results_store = zeros(number_runs,length(depth),length(numTree),length(numSplits),length(classifierId),length(bagSizes),length(decChoice));
train_time_store = zeros(number_runs,length(depth),length(numTree),length(numSplits),length(classifierId),length(bagSizes),length(decChoice));
test_time_store = zeros(number_runs,length(depth),length(numTree),length(numSplits),length(classifierId),length(bagSizes),length(decChoice));

forest_options = struct;
forest_options.verbose = false; % outputs training update
forest_options.classifierCommitFirst = false;
for depth_idx = 1:length(depth)
    forest_options.depth = depth(depth_idx);
    for numTree_idx = 1:length(numTree)
        forest_options.numTrees = numTree(numTree_idx);
        for numSplits_idx = 1:length(numSplits)
            forest_options.numSplits = numSplits(numSplits_idx);
            for classifierId_idx = 1:length(classifierId)
                forest_options.classifierId = classifierId(classifierId_idx);
                for bagSizes_idx = 1:length(bagSizes)
                    forest_options.bagSizes = bagSizes(bagSizes_idx);
                    for decChoice_idx = 1:length(decChoice)
                        for run_idx = 1:number_runs
                            forest_options.decChoice = decChoice_idx(decChoice_idx);
                            % train the forest
                            tic
                            forest_model = forestTrain(data_train,true_class_vector,forest_options);
                            train_time = toc;
                            train_time_store(run_idx,depth_idx,numTree_idx,numSplits_idx,classifierId_idx,bagSizes_idx,decChoice_idx) = train_time;
                            tic
                            predicted_class_vector = forestTest(forest_model,data_test);
                            test_time = toc;
                            test_time_store(run_idx,depth_idx,numTree_idx,numSplits_idx,classifierId_idx,bagSizes_idx,decChoice_idx) = test_time;
                            % results
                            results = sum(~logical(true_class_vector-predicted_class_vector))/length(true_class_vector)
                            results_store(run_idx,depth_idx,numTree_idx,numSplits_idx,classifierId_idx,bagSizes_idx,decChoice_idx) = results;
                        end
                    end
                end
            end
        end
    end
end
save('Q2_sweep_coarse_entr.mat','results_store');