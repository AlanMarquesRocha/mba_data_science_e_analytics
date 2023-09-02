clear
clc

% Importando a base de dados:
data_base = readmatrix('data.dat');

% Excluindo as colunas 26 e 32 da matriz data_base
data_base(:, [26, 32]) = [];

% Separando atributos e classes da base de dados:
x = data_base(:, 1:34); % atributos [colunas 1 a 34]
y = data_base(:, 35);   % classes [coluna 35]

% Normalização dos dados pelo z-score
x_norm = zscore(x);

% Atribuindo os atributos normalizados à variável X_norm:
X_norm = x_norm;

% Definindo o número de partes para a validação cruzada
k = 5;

% Inicializando as variáveis para armazenar as métricas de desempenho
accuracy = zeros(k, 1);
precision = zeros(k, 1);
sensitivity = zeros(k, 1);
f1_score = zeros(k, 1);

% Realizando a validação cruzada
cv = cvpartition(y, 'KFold', k);

for i = 1:k
    % Dividindo os dados em conjunto de treinamento e teste
    train_idx = training(cv, i);
    test_idx = test(cv, i);
    
    X_train = X_norm(train_idx, :);
    y_train = y(train_idx);
    
    X_test = X_norm(test_idx, :);
    y_test = y(test_idx);
    
    % Treinando o modelo MLP
    mdl = fitnet(3); % Exemplo: 3 neurônios na camada oculta
    mdl = train(mdl, X_train', y_train');
    
    % Fazendo previsões no conjunto de teste
    y_pred = round(sim(mdl, X_test'))';
    
    % Calculando as métricas de desempenho
    accuracy(i) = sum(y_pred == y_test) / length(y_test);
    tp = sum(y_pred == 1 & y_test == 1);
    fp = sum(y_pred == 1 & y_test == 0);
    tn = sum(y_pred == 0 & y_test == 0);
    fn = sum(y_pred == 0 & y_test == 1);
    precision(i) = tp / (tp + fp);
    sensitivity(i) = tp / (tp + fn);
    f1_score(i) = 2 * (precision(i) * sensitivity(i)) / (precision(i) + sensitivity(i));
    
    % Plotagem da matriz de confusão
    figure;
    confusion_matrix = confusionmat(y_test, y_pred);
    class_labels = unique(y); % Obtendo os rótulos das classes
    title_str = sprintf('Matriz de Confusão para k = %d', i);
    heatmap(class_labels, class_labels, confusion_matrix, ...
        'Title', title_str, 'XLabel', 'Valor predito', 'YLabel', 'Valor verdadeiro');
    colorbar;
end

% Calculando as médias das métricas de desempenho
mean_accuracy = mean(accuracy);
mean_precision = mean(precision);
mean_sensitivity = mean(sensitivity);
mean_f1_score = mean(f1_score);

% Exibindo as métricas de desempenho
fprintf('Acurácia média: %.4f\n', mean_accuracy);
fprintf('Precisão média: %.4f\n', mean_precision);
fprintf('Sensibilidade média: %.4f\n', mean_sensitivity);
fprintf('F1-Score médio: %.4f\n', mean_f1_score);
