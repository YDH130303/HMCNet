%% 初始化环境
clear; clc; close all;
rng(0); % 固定随机种子

%% 参数设置
params.patchSize = 8;        % 图像块尺寸
params.batchSize = 32;       % 批处理大小
params.numEpochs = 50;       % 训练轮次
params.lr = 1e-4;            % 学习率
params.embedDim = 64;        % 嵌入维度
params.plotFreq = 50;        % 可视化频率
params.photonLevels = 100:200:1000; 
params.electronicStds = 0.01:0.02:0.10;
params.numTrainPatches = 20000; % 训练样本量

%% 加载数据集 - 修复1：直接使用原始datastore
imds = imageDatastore('E:\lung\normal', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.jpg','.png','.tif'});

% 获取图像数量
numImages = numel(imds.Files);
fprintf('数据集包含 %d 张图像\n', numImages);

%% 混合噪声函数 (泊松+高斯)
function noisy_img = add_mixed_noise(img, photon_level, electronic_sigma)
    scaled_img = img * photon_level;
    noisy_img = poissrnd(scaled_img);
    noisy_img = noisy_img + electronic_sigma * randn(size(scaled_img)) * photon_level;
    noisy_img = noisy_img / photon_level;
    noisy_img = max(min(noisy_img, 1), 0);
end

%% 修复的网络结构
function net = createTransformerNet(inputSize, embedDim)
    net = layerGraph();
    
    % 1. 输入层
    input = imageInputLayer(inputSize, 'Name','input','Normalization','none');
    net = addLayers(net, input);
    
    % 2. 初始特征提取
    convInit = convolution2dLayer(3, embedDim, 'Padding','same', 'Name','conv_init');
    bnInit = batchNormalizationLayer('Name','bn_init');
    reluInit = reluLayer('Name','relu_init');
    
    net = addLayers(net, convInit);
    net = addLayers(net, bnInit);
    net = addLayers(net, reluInit);
    net = connectLayers(net, 'input', 'conv_init');
    net = connectLayers(net, 'conv_init', 'bn_init');
    net = connectLayers(net, 'bn_init', 'relu_init');
    
    % 3. 自注意力分支 - 修复2：调整通道数
    qChannels = 16; % 原为 embedDim/4，改为固定值避免不整除
    kChannels = 16;
    vChannels = 32;
    
    convQ = convolution2dLayer(1, qChannels, 'Padding','same', 'Name','conv_Q');
    convK = convolution2dLayer(1, kChannels, 'Padding','same', 'Name','conv_K');
    convV = convolution2dLayer(1, vChannels, 'Padding','same', 'Name','conv_V');
    
    net = addLayers(net, convQ);
    net = addLayers(net, convK);
    net = addLayers(net, convV);
    net = connectLayers(net, 'relu_init', 'conv_Q');
    net = connectLayers(net, 'relu_init', 'conv_K');
    net = connectLayers(net, 'relu_init', 'conv_V');
    
    % 4. 自注意力计算层
    attnLayer = functionLayer(@(X) computeAttention(X, qChannels, kChannels, vChannels), 'Name','attention');
    net = addLayers(net, attnLayer);
    
    % 5. 前馈网络
    convFF1 = convolution2dLayer(1, 128, 'Padding','same', 'Name','conv_ff1');
    bnFF1 = batchNormalizationLayer('Name','bn_ff1');
    reluFF1 = reluLayer('Name','relu_ff1');
    convFF2 = convolution2dLayer(1, embedDim, 'Padding','same', 'Name','conv_ff2');
    bnFF2 = batchNormalizationLayer('Name','bn_ff2');
    
    net = addLayers(net, convFF1);
    net = addLayers(net, bnFF1);
    net = addLayers(net, reluFF1);
    net = addLayers(net, convFF2);
    net = addLayers(net, bnFF2);
    
    % 6. 残差连接
    addResidual = additionLayer(2, 'Name','add_residual');
    net = addLayers(net, addResidual);
    
    % 7. 修复的输出层：添加sigmoid激活
    convFinal = convolution2dLayer(3, 1, 'Padding','same', 'Name','conv_final');
    sigmoidFinal = sigmoidLayer('Name','sigmoid_final');
    regressionOut = regressionLayer('Name','output');
    
    net = addLayers(net, convFinal);
    net = addLayers(net, sigmoidFinal);
    net = addLayers(net, regressionOut);
    
    % 连接所有层
    concatLayer = depthConcatenationLayer(3, 'Name', 'concat_QKV');
    net = addLayers(net, concatLayer);
    
    net = connectLayers(net, 'conv_Q', 'concat_QKV/in1');
    net = connectLayers(net, 'conv_K', 'concat_QKV/in2');
    net = connectLayers(net, 'conv_V', 'concat_QKV/in3');
    net = connectLayers(net, 'concat_QKV', 'attention');
    net = connectLayers(net, 'attention', 'conv_ff1');
    net = connectLayers(net, 'conv_ff1', 'bn_ff1');
    net = connectLayers(net, 'bn_ff1', 'relu_ff1');
    net = connectLayers(net, 'relu_ff1', 'conv_ff2');
    net = connectLayers(net, 'conv_ff2', 'bn_ff2');
    net = connectLayers(net, 'bn_ff2', 'add_residual/in1');
    net = connectLayers(net, 'relu_init', 'add_residual/in2');
    net = connectLayers(net, 'add_residual', 'conv_final');
    net = connectLayers(net, 'conv_final', 'sigmoid_final');
    net = connectLayers(net, 'sigmoid_final', 'output');
end

%% 修复的自定义注意力函数
function Y = computeAttention(X, qChannels, kChannels, vChannels)
    % 获取输入尺寸
    [h, w, totalChannels, b] = size(X);
    
    % 验证通道数
    expectedChannels = qChannels + kChannels + vChannels;
    if totalChannels ~= expectedChannels
        error('输入通道数 (%d) 与预期 (%d) 不匹配', totalChannels, expectedChannels);
    end
    
    % 分割特征图
    Q = X(:, :, 1:qChannels, :);
    K = X(:, :, qChannels+1:qChannels+kChannels, :);
    V = X(:, :, qChannels+kChannels+1:end, :);
    
    % 初始化输出
    Y = zeros(h, w, vChannels, b, 'like', X);
    
    % 逐批次计算注意力
    for i = 1:b
        % 展平空间维度
        Q_flat = reshape(Q(:,:,:,i), [h*w, qChannels]);
        K_flat = reshape(K(:,:,:,i), [h*w, kChannels]);
        V_flat = reshape(V(:,:,:,i), [h*w, vChannels]);
        
        % 计算注意力分数
        attn_scores = Q_flat * K_flat';
        attn_scores = attn_scores / sqrt(kChannels);
        
        % 手动实现softmax
        max_scores = max(attn_scores, [], 2);
        exp_scores = exp(attn_scores - max_scores);
        sum_exp = sum(exp_scores, 2);
        attn_weights = exp_scores ./ sum_exp;
        
        % 应用注意力到值
        attn_output = attn_weights * V_flat;
        
        % 恢复空间维度
        attn_output = reshape(attn_output, [h, w, vChannels]);
        Y(:,:,:,i) = attn_output;
    end
end

% 创建网络
net = createTransformerNet([params.patchSize params.patchSize 1], params.embedDim);
analyzeNetwork(net);

%% 修复的预生成训练数据函数 - 修复3：直接使用原始datastore
function [X, Y] = generateTrainingData(imds, params, numPatches)
    % 初始化输出数组
    X = zeros(params.patchSize, params.patchSize, 1, numPatches);
    Y = zeros(params.patchSize, params.patchSize, 1, numPatches);
    
    % 预处理函数
    preprocess = @(x) im2double(imresize(im2gray(x), [128 128]));
    
    for p = 1:numPatches
        % 随机选择图像
        imgIdx = randi(numel(imds.Files));
        img = readimage(imds, imgIdx);
        
        % 预处理
        img = preprocess(img);
        
        % 确保单通道灰度图
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        img = im2double(img);
        
        % 生成随机噪声参数
        photon_level = params.photonLevels(randi(numel(params.photonLevels)));
        electronic_sigma = params.electronicStds(randi(numel(params.electronicStds)));
        
        % 添加混合噪声
        noisy_img = add_mixed_noise(img, photon_level, electronic_sigma);
        
        % 随机裁剪图像块
        [h, w] = size(img);
        i = randi(h - params.patchSize + 1);
        j = randi(w - params.patchSize + 1);
        
        % 提取图像块并赋值
        clean_patch = img(i:i+params.patchSize-1, j:j+params.patchSize-1);
        noisy_patch = noisy_img(i:i+params.patchSize-1, j:j+params.patchSize-1);
        
        Y(:, :, 1, p) = clean_patch;
        X(:, :, 1, p) = noisy_patch;
    end
end

% 生成训练数据 - 使用原始imds
[X_train, Y_train] = generateTrainingData(imds, params, params.numTrainPatches);
fprintf('训练数据生成完成：%d 个图像块\n', params.numTrainPatches);



%% 显示训练数据图像块（4行每行10个）- 紧凑布局
% 创建新图窗
figure('Name', '训练图像块样本 - 紧凑布局', 'Position', [100, 100, 1200, 500]);

% 使用tiledlayout创建紧凑布局
t = tiledlayout(4, 10, 'TileSpacing', 'none', 'Padding', 'tight');

% 随机选择40个样本索引
sampleIndices = randperm(params.numTrainPatches, 40);

% 显示噪声图像块（上半部分：2行）
for i = 1:20
    nexttile;
    img = X_train(:, :, 1, sampleIndices(i));
    imshow(img);
    title(sprintf('N-%d', i), 'FontSize', 8);
end

% 显示干净图像块（下半部分：2行）
for i = 1:20
    nexttile;
    img = Y_train(:, :, 1, sampleIndices(i));
    imshow(img);
    title(sprintf('C-%d', i), 'FontSize', 8);
end

% 添加整体标题和说明
title(t, '训练数据样本展示 (上2行: 噪声图像块 | 下2行: 干净图像块)', 'FontSize', 12);
annotation('textbox', [0.02, 0.48, 0.1, 0.05], 'String', '干净图像块', ...
           'EdgeColor', 'none', 'FontWeight', 'bold', 'Color', 'g', 'FontSize', 10);
annotation('textbox', [0.02, 0.98, 0.1, 0.05], 'String', '噪声图像块', ...
           'EdgeColor', 'none', 'FontWeight', 'bold', 'Color', 'r', 'FontSize', 10);


%% 训练配置
options = trainingOptions('adam',...
    'MaxEpochs', params.numEpochs,...
    'MiniBatchSize', params.batchSize,...
    'Plots', 'training-progress',...
    'Verbose', true,...
    'LearnRateSchedule', 'piecewise',...
    'LearnRateDropFactor', 0.5,...
    'LearnRateDropPeriod', 10,...
    'Shuffle', 'every-epoch',...
    'ExecutionEnvironment', 'gpu');

%% 训练网络
net = trainNetwork(X_train, Y_train, net, options);
save('denoising_network.mat', 'net', 'params');
fprintf('网络训练完成并已保存\n');







%% 测试去噪性能
% 读取并预处理测试图像
testImg = imread('1.jpg');
preprocess = @(x) im2double(imresize(im2gray(x), [128 128]));
testImg = preprocess(testImg); % 使用相同的预处理函数

% 添加噪声
noisyTestImg = add_mixed_noise(testImg, 100, 0.02);

% 分块处理
denoisedImg = zeros(size(testImg));
patchSize = params.patchSize;
for i = 1:patchSize:size(testImg,1)
    for j = 1:patchSize:size(testImg,2)
        end_i = min(i+patchSize-1, size(testImg,1));
        end_j = min(j+patchSize-1, size(testImg,2));
        patch = noisyTestImg(i:end_i, j:end_j);
        
        % 边界填充
        if (end_i-i+1) < patchSize || (end_j-j+1) < patchSize
            padPatch = padarray(patch, [patchSize-(end_i-i+1), patchSize-(end_j-j+1)], 0, 'post');
            denoisedPatch = predict(net, padPatch);
            denoisedPatch = denoisedPatch(1:size(patch,1), 1:size(patch,2));
        else
            denoisedPatch = predict(net, patch);
        end
        
        denoisedImg(i:end_i, j:end_j) = denoisedPatch;
    end
end

%% 修复评估指标计算
mseNoisy = mean((testImg(:) - noisyTestImg(:)).^2);
psnrNoisy = psnr(noisyTestImg, testImg);
ssimNoisy = ssim(noisyTestImg, testImg);

mseDenoised = mean((testImg(:) - denoisedImg(:)).^2);
psnrDenoised = psnr(denoisedImg, testImg);
ssimDenoised = ssim(denoisedImg, testImg);

%% 可视化结果
figure('Position', [100 100 1200 400])
subplot(131), imshow(testImg), title('原始图像')
subplot(132), imshow(noisyTestImg), 
title(sprintf('噪声图像\nMSE:%.4f, PSNR:%.2fdB, SSIM:%.4f',...
    mseNoisy, psnrNoisy, ssimNoisy))
subplot(133), imshow(denoisedImg), 
title(sprintf('去噪结果\nMSE:%.4f, PSNR:%.2fdB, SSIM:%.4f',...
    mseDenoised, psnrDenoised, ssimDenoised))

%% 保存结果
save('denoising_results.mat', 'testImg', 'noisyTestImg', 'denoisedImg', ...
    'mseNoisy', 'psnrNoisy', 'ssimNoisy', 'mseDenoised', 'psnrDenoised', 'ssimDenoised');
disp('去噪完成！结果已保存');