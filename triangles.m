%Load the images into matlab

dataFolder = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
imageFolderTrain = fullfile(dataFolder,'trainingImages');
labelFolderTrain = fullfile(dataFolder,'trainingLabels');

%Create image data store for the images
imdsTrain = imageDatastore(imageFolderTrain);

%Create a pixelLabelDatastore for the ground truth pixel labels.
classNames = ["triangle" "background"];
labels = [255 0];
pxdsTrain = pixelLabelDatastore(labelFolderTrain,classNames,labels);

%Create pixel label datastore and count the pixels in each label
pximdsTrain = pixelLabelImageDatastore(imdsTrain,pxdsTrain);
tbl = countEachLabel(pximdsTrain);

numberPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / numberPixels;
classWeights = 1 ./ frequency;

%Create network and specify the layers


inputSize = [32 32 1];
filterSize = 3;
numFilters = 32;
numClasses = numel(classNames);

layers = [
    imageInputLayer(inputSize)
    
    convolution2dLayer(filterSize,numFilters,'DilationFactor',1,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(filterSize,numFilters,'DilationFactor',2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(filterSize,numFilters,'DilationFactor',4,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(1,numClasses)
    softmaxLayer
    pixelClassificationLayer('Classes',classNames,'ClassWeights',classWeights)];

    %Specify training options and train the network
    
options = trainingOptions('sgdm', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 64, ... 
    'InitialLearnRate', 1e-3);

%Train the network
net = trainNetwork(pximdsTrain,layers,options);

%Test the network trained using the training data
imageFolderTest = fullfile(dataFolder,'testImages');
imdsTest = imageDatastore(imageFolderTest);
labelFolderTest = fullfile(dataFolder,'testLabels');
pxdsTest = pixelLabelDatastore(labelFolderTest,classNames,labels);

%Make predictions using the test data and trained network.
pxdsPred = semanticseg(imdsTest,net,'WriteLocation',tempdir);


%Evaluate the prediction accuracy using evaluateSemanticSegmentation.
metrics = evaluateSemanticSegmentation(pxdsPred,pxdsTest);

%Read and display the test image triangleTest.jpg.

imgTest = imread('triangleTest.jpg');
figure
imshow(imgTest)
