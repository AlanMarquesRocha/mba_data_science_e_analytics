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

x_balanced = zscore(x_smote);

% Dividindo os dados em conjuntos de treinamento e teste
train_ratio = 0.8; % proporção de dados para treinamento
num_samples = size(x_balanced, 1);
num_train = round(num_samples * train_ratio);
num_test = num_samples - num_train;

% Embaralhando os dados
indices = randperm(num_samples);
x_shuffled = x_balanced(indices, :);
y_shuffled = y_balanced(indices);

% Dividindo em conjuntos de treinamento e teste
x_train = x_shuffled(1:num_train, :);
y_train = y_shuffled(1:num_train);
x_test = x_shuffled(num_train+1:end, :);
y_test = y_shuffled(num_train+1:end);

% Random Forest
num_trees = 120; % número de árvores na floresta

% Criando a floresta de árvores de decisão
forest = TreeBagger(num_trees, x_train, y_train, 'Method', 'classification');

% Realizando a predição no conjunto de teste
y_pred = predict(forest, x_test);

% Convertendo as previsões de células para vetor numérico
y_pred = str2double(y_pred);

% Calculando as métricas de avaliação
accuracy = sum(y_pred == y_test) / numel(y_test);
precision = zeros(1, num_classes);
recall = zeros(1, num_classes);
f1_score = zeros(1, num_classes);

for i = 1:num_classes
    true_positives = sum(y_pred == i & y_test == i);
    false_positives = sum(y_pred == i & y_test ~= i);
    false_negatives = sum(y_pred ~= i & y_test == i);
    
    precision(i) = true_positives / (true_positives + false_positives);
    recall(i) = true_positives / (true_positives + false_negatives);
    f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Calculando a matriz de confusão
confusion_matrix = confusionmat(y_test, y_pred);

% Exibindo a matriz de confusão como uma imagem
figure;
imagesc(confusion_matrix);
colorbar;
title('Matriz de Confusão do Random Forest com SMOTE');
xlabel('Valor predito');
ylabel('Valor verdadeiro');
xticks(1:num_classes);
yticks(1:num_classes);

% Exibindo os valores da matriz de confusão em uma figura separada
figure;
imshow(uint8(zeros(10))); % Ajuste o tamanho da figura conforme necessário
textStrings = num2str(confusion_matrix(:), '%d');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:num_classes);
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
set(gca, 'XTick', 1:num_classes, 'YTick', 1:num_classes);
title('Matriz de Confusão do Random Forest com SMOTE');
xlabel('Valor predito');
ylabel('Valor verdadeiro');


disp(confusion_matrix)