function [ data_train, data_query ] = getData( vocab_size_in )
% Generate training and testing data
showImg = 0; % Show training & testing images and their image feature vector (histogram representation)
PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}
close all;
imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
folderName = './Caltech_101/101_ObjectCategories';
classList = dir(folderName);
classList = {classList(3:end).name} % 10 classes
disp('Loading training images...')
% Load Images -> Description (Dense SIFT)
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
disp('Building visual codebook...')
% Build visual vocabulary (codebook) for 'Bag-of-Words method'
desc_sel = single(vl_colsubset(cat(2,desc_tr{:}), 10e4)); % Randomly select 100k SIFT descriptors for clustering
% K-means clustering
[~, words] = kmeans(desc_sel', vocab_size_in);
disp('Encoding Training Images...')
bags_of_words_training = zeros(vocab_size_in,150); % 150 = 10 classes * 15 images per class
imTrack = 1;
% Vector Quantisation (training images)
for class = 1:10
    for image = 1:15
        image_descriptors = desc_tr(class,image);
        image_descriptors = image_descriptors{1};
        closest_words = knnsearch(words,image_descriptors');
        for k = 1:length(closest_words)
            bags_of_words_training(closest_words(k),imTrack) = bags_of_words_training(closest_words(k),imTrack) + 1;
        end
        imTrack = imTrack + 1;
    end
end
% Clear unused varibles to save memory
clearvars desc_tr desc_sel
if showImg
    figure('Units','normalized','Position',[.05 .1 .4 .9]);
    suptitle('Test image samples');
end
disp('Processing testing images...');
cnt = 1;
% Load Images -> Description (Dense SIFT)
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
disp('Encoding testing images');
if showImg
    figure('Units','normalized','Position',[.5 .1 .4 .9]);
    suptitle('Testing image representations: 256-D histograms');
end
% Quantisation
bags_of_words_testing = zeros(vocab_size_in,150);
imTrack = 1;
% Vector Quantisation
for class = 1:10
    for image = 1:15
        image_descriptors = desc_te(class,image);
        image_descriptors = image_descriptors{1};
        closest_words = knnsearch(words,image_descriptors');
        for k = 1:length(closest_words)
            bags_of_words_testing(closest_words(k),imTrack) = bags_of_words_testing(closest_words(k),imTrack) + 1;
        end
        imTrack = imTrack + 1;
    end
end
data_train = bags_of_words_training';
data_query = bags_of_words_testing';
end