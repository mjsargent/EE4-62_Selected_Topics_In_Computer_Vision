function [descVector, classVector, imageIdx]  = getDescVector(descriptors_in)
    descVector = [];
    classVector = [];
    %imageIdx = [];
    imageIdx = zeros(151,1);
    for class = 1:10
        for image = 1:15
            idx = image + (class-1)*15;
            imageIdx(idx) = length(descVector)+1;
            descVector = [descVector; descriptors_in{class,image}'];
            classVector = [classVector;ones(size(descriptors_in{class,image},2),1)*class];
            %imageIdx = [imageIdx;ones(size(descriptors_in{class,image},2),1)*idx];
        end
    end
    imageIdx(151) = length(descVector)+1;
end