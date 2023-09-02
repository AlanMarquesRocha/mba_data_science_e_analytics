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


% Definindo o número de árvores na floresta
num_arvores = 200;

% Definindo o número de folds na validação cruzada
k = 5;


% Inicializando variáveis para armazenar as métricas de desempenho
acuracias = zeros(k, 1);
matrizes_confusao = cell(k, 1);

% Aplicando a validação cruzada estratificada
cv = cvpartition(y, 'KFold', k);

% Inicializando variáveis para armazenar as métricas de desempenho
precisoes = zeros(k, 1);
sensibilidades = zeros(k, 1);
f1_scores = zeros(k, 1);

for i = 1:k
    % Separando os dados de treinamento e teste para a fold atual
    dados_treinamento = X_norm(training(cv, i), :);
    rotulos_treinamento = y(training(cv, i), :);
    dados_teste = X_norm(test(cv, i), :);
    rotulos_teste = y(test(cv, i), :);
    
    % Treinando o modelo de Random Forest
    modelo = TreeBagger(num_arvores, dados_treinamento, rotulos_treinamento);
    
    % Realizando as previsões nos dados de teste
    previsoes = predict(modelo, dados_teste);
    previsoes = str2double(previsoes);
    
    % Calculando a acurácia da fold atual
    acuracia = sum(previsoes == rotulos_teste) / numel(rotulos_teste);
    acuracias(i) = acuracia;
    
    % Calculando a matriz de confusão da fold atual
    matriz_confusao = confusionmat(rotulos_teste, previsoes);
    matrizes_confusao{i} = matriz_confusao;

    % Calculando as métricas de desempenho da fold atual
    TP = matriz_confusao(2, 2);  % True Positives
    TN = matriz_confusao(1, 1);  % True Negatives
    FP = matriz_confusao(1, 2);  % False Positives
    FN = matriz_confusao(2, 1);  % False Negatives
    
    precisao = TP / (TP + FP);
    sensibilidade = TP / (TP + FN);
    f1_score = 2 * (precisao * sensibilidade) / (precisao + sensibilidade);
    
    % Armazenando as métricas da fold atual
    precisoes(i) = precisao;
    sensibilidades(i) = sensibilidade;
    f1_scores(i) = f1_score;

end

% Calculando a média das acurácias
media_acuracias = mean(acuracias);

% Exibindo a média das acurácias e as matrizes de confusão
disp('Acurácia média:');
disp(media_acuracias);

disp('Matrizes de Confusão:');
for i = 1:k
    disp(['Fold ', num2str(i)]);
    disp(matrizes_confusao{i});
end

% Plotando o histograma da média de cada atributo
%figure;
%for i = 1:size(X_norm, 2)
%    subplot(6, 6, i); % 6x6 subplot grid
%    histogram(X_norm(:, i), 'FaceColor', 'b', 'EdgeColor', 'k');
%    legend('Frequência');
%    title(['Atributo ', num2str(i)]);
%end

% Calculando as médias das métricas de desempenho
media_precisao = mean(precisoes);
media_sensibilidade = mean(sensibilidades);
media_f1_score = mean(f1_scores);

% Exibindo as métricas de desempenho médias
disp('Métricas de Desempenho Médias:');
disp(['Precisão: ', num2str(media_precisao)]);
disp(['Sensibilidade: ', num2str(media_sensibilidade)]);
disp(['F1-score: ', num2str(media_f1_score)]);
