clear
clc

% Importando a base de dados:
data_base = readmatrix('data.dat');

% Excluindo as colunas 26 e 32 da matriz data_base
data_base(:, [26, 32]) = [];

% Separando atributos e classes da base de dados:
x = data_base(:, 1:34); % atributos [colunas 1 a 34]
y = data_base(:, 35);   % classes [coluna 35]

% Verificando o desbalanceamento das classes
class_counts = histcounts(y);
fprintf('Número de amostras por classe:\n');
for i = 1:numel(class_counts)
    fprintf('Classe %d: %d amostras\n', i, class_counts(i));
end

% Determinando o número de amostras para igualar as classes
max_samples = max(class_counts);
num_classes = max(y);
num_samples_per_class = ones(1, num_classes) * max_samples;

% Aplicando o SMOTE para balancear as classes
smote_ratio = 1; % Defina o valor de oversampling desejado
smote_neighbors = 5; % Defina o número de vizinhos para gerar amostras sintéticas

smote_data = [];
smote_labels = [];
for class_label = 1:num_classes
    class_samples = x(y == class_label, :);
    num_samples = sum(y == class_label);
    
    if num_samples < max_samples
        num_to_generate = max_samples - num_samples;
        synthetic_samples = [];
        
        for j = 1:num_to_generate
            random_idx = randi(num_samples);
            nn_indices = knnsearch(class_samples, class_samples(random_idx, :), 'K', smote_neighbors+1);
            nn_indices = nn_indices(2:end);
            
            nn_sample = class_samples(nn_indices(randi(smote_neighbors)), :);
            synthetic_sample = class_samples(random_idx, :) + rand(1, size(class_samples, 2)) .* (nn_sample - class_samples(random_idx, :));
            
            synthetic_samples = [synthetic_samples; synthetic_sample];
        end
        
        smote_data = [smote_data; synthetic_samples];
        smote_labels = [smote_labels; repmat(class_label, num_to_generate, 1)];
    end
end

% Concatenando os dados sintéticos com os dados originais
x_smote = [x; smote_data];
y_balanced = [y; smote_labels];

% Realizando a normalização dos dados sintéticos com a técnica z-score
x_balanced = zscore(x_smote);

% Divisão dos conjuntos de treinamento, validação e teste
rng(0.1); % Define a semente para reproducibilidade
indices = randperm(size(x_balanced, 1));
train_size = round(0.8 * size(x_balanced, 1));
val_size = round(0.1 * size(x_balanced, 1));

x_train = x_balanced(indices(1:train_size), :);
y_train = y_balanced(indices(1:train_size));

x_val = x_balanced(indices(train_size+1:train_size+val_size), :);
y_val = y_balanced(indices(train_size+1:train_size+val_size));

x_test = x_balanced(indices(train_size+val_size+1:end), :);
y_test = y_balanced(indices(train_size+val_size+1:end));

% Treinamento do modelo de árvore de decisão
tree = fitctree(x_train, y_train);

% Predição nos conjuntos de treinamento, validação e teste
y_train_pred = predict(tree, x_train);
y_val_pred = predict(tree, x_val);
y_test_pred = predict(tree, x_test);

% Cálculo da acurácia
train_accuracy = sum(y_train_pred == y_train) / numel(y_train);
val_accuracy = sum(y_val_pred == y_val) / numel(y_val);
test_accuracy = sum(y_test_pred == y_test) / numel(y_test);

% Cálculo da precisão
train_precision = sum(y_train_pred == y_train & y_train_pred == 1) / sum(y_train_pred == 1);
val_precision = sum(y_val_pred == y_val & y_val_pred == 1) / sum(y_val_pred == 1);
test_precision = sum(y_test_pred == y_test & y_test_pred == 1) / sum(y_test_pred == 1);

% Cálculo da Sensibilidade ou recall
train_recall = sum(y_train_pred == y_train & y_train_pred == 1) / sum(y_train == 1);
val_recall = sum(y_val_pred == y_val & y_val_pred == 1) / sum(y_val == 1);
test_recall = sum(y_test_pred == y_test & y_test_pred == 1) / sum(y_test == 1);

% Cálculo da F1-Score
train_f1_score = 2 * (train_precision * train_recall) / (train_precision + train_recall);
val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall);
test_f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall);

fprintf('Métricas de avaliação:\n');
fprintf('Conjunto de Treinamento:\n');
fprintf('Acurácia: %.4f\n', train_accuracy);
fprintf('Precisão: %.4f\n', train_precision);
fprintf('Sensibilidade: %.4f\n', train_recall);
fprintf('F1-score: %.4f\n', train_f1_score);
fprintf('\n');
fprintf('Conjunto de Validação:\n');
fprintf('Acurácia: %.4f\n', val_accuracy);
fprintf('Precisão: %.4f\n', val_precision);
fprintf('Sensibilidade: %.4f\n', val_recall);
fprintf('F1-score: %.4f\n', val_f1_score);
fprintf('\n');
fprintf('Conjunto de Teste:\n');
fprintf('Acurácia: %.4f\n', test_accuracy);
fprintf('Precisão: %.4f\n', test_precision);
fprintf('Sensibilidade: %.4f\n', test_recall);
fprintf('F1-score: %.4f\n', test_f1_score);

% Calcula a matriz de confusão para o conjunto de teste
C = confusionmat(y_test, y_test_pred);

% Exibe a matriz de confusão
fprintf('Matriz de Confusão (Conjunto de Teste):\n');
disp(C);
