% q3 coarse sweep
clear all; close all; 
warning('off','all');
init;clc;
rng(1)
showImg = 0; % Show training & testing images and their image feature vector (histogram representation)
PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}
imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
folderName = './Caltech_101/101_ObjectCategories';
classList = dir(folderName);
classList = {classList(3:end).name}; % 10 classes
% Load Images -> Description (Dense SIFT)
disp('Loading training images...')
cnt = 1;
if showImg
    figure('Units','normalized','Position',[.05 .1 .4 .9]);
    suptitle('Training image samples');
end
for c = 1:length(classList)
    subFolderName = fullfile(folderName,classList{c});
    imgList = dir(fullfile(subFolderName,'*.jpg'));
    imgIdx{c} = randperm(length(imgList));
    imgIdx_tr = imgIdx{c}(1:imgSel(1));
    imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
    for i = 1:length(imgIdx_tr)
        I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
        % Visualise
        if i < 6 & showImg
            subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
            imshow(I);
            cnt = cnt+1;
            drawnow;
        end
        if size(I,3) == 3
            I = rgb2gray(I); % PHOW work on gray scale image
        end
        % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
        [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
    end
end
% Load Images -> Description (Dense SIFT)
% disp('Processing testing images...');
cnt = 1;
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
            end
        end
train_idxs = cell(10,15);
for class = 1:10
    for image = 1:15
        train_idxs{class,image} = class*ones(1,size(desc_tr{class,image},2));
    end
end
train_idxs = cat(2,train_idxs{:});
[desc_sel,t_ind] = vl_colsubset(cat(2,desc_tr{:}), 10e4); % Randomly select 100k SIFT descriptors for clustering
% Build visual vocabulary (codebook) for 'Bag-of-Words method'

% optain Desc 
[trainingDescVec, trainingDescClass, trainingImageIdx] = getDescVector(desc_tr);
[testingDescVec, testingDescClass, ~] = getDescVector(desc_te);
train_sample = randperm(length(trainingDescVec),100000);

%% loop options
c_depths = [6,7,8,9,10];%; % 2,5,8 % 8,10 % [8,10]
c_numTrees = [10];%%[10,20]; % 5,10,100 % 10,20 % [5,20,40]
c_numSplits = [100];%[100,200]; % 20,100 % 100,200 % [20,100]%
c_bagSizes = [150000]; % 10000,100000 % 100000

codebook_options = struct;
codebook_options.verbose = true; % outputs training update
codebook_options.classifierId = 1;
codebook_options.classifierCommitFirst = false;
codebook_options.decChoice = 1;

f_depths = [7]; % 3,7 %7,9 % 6,8
f_numTrees = [1000]; % 100, 500 % 500,1000
f_numSplits = [150]; % 20,100 % 100,150
f_bagSizes = [150];% 50, 150 % 150

forest_options = struct;
forest_options.verbose = true; % outputs training update
forest_options.classifierId = 1;
forest_options.classifierCommitFirst = false;
forest_options.decChoice = 1;
true_class_vector = reshape((ones(10,15).*[1:10]')',[150,1]);
%%
numLoops = 10;
results_store = zeros(length(c_depths),length(c_numTrees),length(c_numSplits),length(c_bagSizes),...
                 length(f_depths),length(f_numTrees),length(f_numSplits),length(f_bagSizes),numLoops);
train_time_store = zeros(length(c_depths),length(c_numTrees),length(c_numSplits),length(c_bagSizes),numLoops);
quant_time_store = zeros(length(c_depths),length(c_numTrees),length(c_numSplits),length(c_bagSizes),numLoops);
for loop_idx = 1:numLoops
for cd_idx = 1:length(c_depths)
    cd_idx;
    codebook_options.depth = c_depths(cd_idx);
    for ct_idx = 1:length(c_numTrees)
        ct_idx;
        codebook_options.numTrees = c_numTrees(ct_idx);
        for cn_idx = 1:length(c_numSplits)
            cn_idx;
            codebook_options.numSplits = c_numSplits(cn_idx);
            for cb_idx = 1:length(c_bagSizes)
                cb_idx;
                codebook_options.bagSizes = c_bagSizes(cb_idx);
                
                % train RF codebook
                trainingData = zeros(codebook_options.numTrees*2^(codebook_options.depth-1),150)';
                testingData = zeros(codebook_options.numTrees*2^(codebook_options.depth-1),150)';
                %codebook_model = forestTrain(trainingDescVec(train_sample,:),trainingDescClass(train_sample),codebook_options);
                tic
                codebook_model = forestTrain(desc_sel',train_idxs(t_ind)',codebook_options);
                time = toc;
                train_time_store(cd_idx,ct_idx,cn_idx,cb_idx,loop_idx) = time;
                tic
                for class = 1:10
                    for image = 1:15
                        idx = (class-1)*15 + image;
                        trainingData(idx,:) = findImageHistogram(codebook_model, desc_tr{class,image}', codebook_options);
                        testingData(idx,:) = findImageHistogram(codebook_model, desc_te{class,image}', codebook_options);
                    end
                end
                time = toc;
                quant_time_store(cd_idx,ct_idx,cn_idx,cb_idx,loop_idx) = time;
                for fd_idx = 1:length(f_depths)
                    forest_options.depth = f_depths(fd_idx);
                    for ft_idx = 1:length(f_numTrees)
                        forest_options.numTrees = f_numTrees(ft_idx);
                        for fn_idx = 1:length(f_numSplits)
                            forest_options.numSplits = f_numSplits(fn_idx);
                            for fb_idx = 1:length(f_bagSizes)
                                forest_options.bagSizes = f_bagSizes(fb_idx);

                                % train the forest
                                forest_model = forestTrain(trainingData,true_class_vector,forest_options);
                                predicted_class_vector = forestTest(forest_model,testingData);
                                % results
                                results = sum(~logical(true_class_vector-predicted_class_vector))/length(true_class_vector)
                                results_store(cd_idx,ct_idx,cn_idx,cb_idx,fd_idx,ft_idx,fn_idx,fb_idx,loop_idx) = results;
                            end
                        end
                    end
                end
                save('q3_fix_vary_cdepth.mat','results_store','train_time_store','quant_time_store')
            end
        end
    end
end
end