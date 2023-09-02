clear
clc

% Importando a base de dados:
data_base = readmatrix('data.dat');

% Excluindo as colunas 26 e 32 da matriz data_base
data_base(:, [26, 32]) = [];

% Separando atributos e classes da base de dados:
x = data_base(:, 1:34); % atributos [colunas 1 a 34]
y = data_base(:, 35);   % classes [coluna 35]

% Normalizando os dados para ficarem entre 0 e 1
%X_norm = (x - min(x)) ./ (max(x) - min(x));

% Realizando a normalização dos dados através de zscore.
% Calculando a média e desvio padrão de cada coluna dos atributos:
atrib_medias = mean(x);
atrib_desv_padrao = std(x);

% Subtraindo a média de cada coluna dos atributos e dividindo
% pelo desvio padrão.
atrib_norm = (x - atrib_medias) ./ atrib_desv_padrao;
% O operador ./ é usado para realizar a divisão elemento a elemento entre
% dois vetores ou matrizes.

% Atribuindo os atributos normalizados a variável X:
X_norm = atrib_norm;

% Implementação da Árvore de Decisão com validação cruzada estratificada para k = 5:
k = 5;
cv_acc = zeros(1, k);
cv_precision = zeros(1, k);
cv_recall = zeros(1, k);
cv_f1_score = zeros(1, k);

indices = crossvalind('Kfold', y, k);

for i = 1:k
    test_indices = (indices == i);
    train_indices = ~test_indices;
    
    X_train = X_norm(train_indices, :);
    y_train = y(train_indices);
    
    X_test = X_norm(test_indices, :);
    y_test = y(test_indices);
    
    % Mapeando os rótulos de -1, 0 e 1 para 0, 1 e 2
    y_train_mapped = y_train + 1;
    y_test_mapped = y_test + 1;
    
    % Treinamento da Árvore de Decisão:
    tree = fitctree(X_train, y_train_mapped);
    
    % Predição dos dados de teste:
    y_pred_mapped = predict(tree, X_test);
    
    % Mapeando os rótulos de volta para -1, 0 e 1
    y_pred = y_pred_mapped - 1;
    
    % Cálculo das métricas:
    accuracy = sum(y_pred == y_test) / numel(y_test);
    precision = sum(y_pred == 1 & y_test == 1) / sum(y_pred == 1);
    recall = sum(y_pred == 1 & y_test == 1) / sum(y_test == 1);
    f1_score = 2 * (precision * recall) / (precision + recall);
    
    cv_acc(i) = accuracy;
    cv_precision(i) = precision;
    cv_recall(i) = recall;
    cv_f1_score(i) = f1_score;
    
    % Plotagem da matriz de confusão
    figure;
    confusion_matrix = confusionmat(y_test, y_pred);
    class_labels = [-1, 0, 1]; % Rótulos corrigidos
    title_str = sprintf('Matriz de Confusão para k = %d', i);
    heatmap(class_labels, class_labels, confusion_matrix, ...
        'Title', title_str, 'XLabel', 'Valor predito', 'YLabel', 'Valor verdadeiro');
    colorbar;
end

% Calculando a média de cada métrica
mean_acc = mean(cv_acc);
mean_precision = mean(cv_precision);
mean_recall = mean(cv_recall);
mean_f1_score = mean(cv_f1_score);

% Apresentação das métricas:
fprintf('Acurácia média: %.4f\n', mean_acc);
fprintf('Precisão média: %.4f\n', mean_precision);
fprintf('Sensibilidade média: %.4f\n', mean_recall);
fprintf('F1-score médio: %.4f\n', mean_f1_score);
