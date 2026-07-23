clear; clc;

% ======================
% 1. 读取配置
% ======================
cfg = config();

% ======================
% 2. 读取数据
% ======================
data = readmatrix(cfg.excel_path);

% ======================
% 3. 计算pBRDF
% ======================
dop = compute_pbrdf(data, cfg);

% ======================
% 4. 输出结果
% ======================
plot(dop);
title('DOP vs Wavelength');
xlabel('Index');
ylabel('DOP');