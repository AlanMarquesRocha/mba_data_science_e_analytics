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

% Realizando a normalização dos atributos para o intervalo [1, 2]
x_normalized = zscore(x_smote);

% Convertendo as classes para valores inteiros positivos
unique_classes = unique(y_balanced);
class_mapping = containers.Map(unique_classes, 1:length(unique_classes));
y_integers = zeros(size(y_balanced));
for i = 1:length(y_balanced)
    y_integers(i) = class_mapping(y_balanced(i));
end

% Convertendo as classes para one-hot encoding usando dummyvar
y_encoded = dummyvar(y_integers);

% [1 3 3] = 80,81%

% Definindo os parâmetros da RNA-MLP
hidden_layer_sizes = [10 10 10]; % Número de neurônios em cada camada oculta
net = fitnet(hidden_layer_sizes); % Criando a RNA-MLP
net.trainParam.showWindow = false; % Não mostrar a janela de treinamento
net.divideParam.trainRatio = 0.8; % Razão de divisão dos dados de treinamento
net.divideParam.valRatio = 0.1; % Razão de divisão dos dados de validação
net.divideParam.testRatio = 0.1; % Razão de divisão dos dados de teste

% Treinando a RNA-MLP
[net, tr] = train(net, x_normalized', y_encoded');

% Realizando a predição dos dados de teste
y_pred = net(x_normalized(tr.testInd,:)');

% Convertendo as saídas preditas para as classes correspondentes
[~, y_pred_labels] = max(y_pred);
y_pred_labels = y_pred_labels';

% Convertendo as classes para os valores originais
y_pred_classes = zeros(size(y_pred_labels));
for i = 1:length(y_pred_labels)
    y_pred_classes(i) = unique_classes(y_pred_labels(i));
end

% Exibindo as métricas de desempenho
fprintf('Métricas do conjunto de teste:\n');

accuracy = sum(y_pred_classes == y_balanced(tr.testInd)) / length(y_balanced(tr.testInd));
fprintf('\nAcurácia: %.2f%%\n', accuracy * 100);

% Calculando a matriz de confusão
C = confusionmat(y_balanced(tr.testInd), y_pred_classes);

% Plotando a matriz de confusão com valores
figure;
imagesc(C);
colorbar;
colormap("parula");
xlabel('Valor predito');
ylabel('Valor verdadeiro');
title('Matriz de Confusão da RNA-MLP com SMOTE');
set(gca, 'XTick', 1:num_classes, 'YTick', 1:num_classes);
xticks(1:num_classes);
yticks(1:num_classes);
xticklabels(unique_classes);
yticklabels(unique_classes);

% Adicionando os valores à matriz de confusão
for i = 1:size(C, 1)
    for j = 1:size(C, 2)
        text(j, i, num2str(C(i, j)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end
end
