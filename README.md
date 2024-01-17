# Ciência de Dados em Produção: Previsão de Vendas + Machine Learning + Bot no Telegram

<img src="https://www.rossmann.cz/-a5435---ib6QCNrH/rossmann-logo-neg-1-jpg?v=a38761" alt="drawing" width="100%"/>

# 1. Projeto Previsão de Vendas Rossmann

## 1.1. Rossmann

Fundada em 1972, Dirk **Rossmann** GmbH é uma das maiores redes de drogarias da Europa, com lojas na Alemanha, Polônia, Hungria, República Tcheca, Turquia, Albânia, Kosovo, Israel e Espanha. Ainda que o foco das drogarias sejam ofertar produtos nas áreas de saúde e beleza, a Rossmann investe em produtos e serviços que fogem desse universo, como ração para animais e serviço fotográfico. 

De acordo com o seu site, a variedade das filiais é norteada pelas necessidades do clientes e a gama de produtos depende do tamanho da loja e localização. A empresa possui atualmente (2023), 60.500 funcionários e 4.514 filiais.

## 1.2. Dados

Os dados foram obtidos no [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) e contém informações sobre as vendas diárias de 1.115 lojas da Rossmann. O conjunto de dados contém informações sobre promoções, feriados e características da loja.

# 2. Contexto - Problema Fictício 

Com o intuito de reformar as lojas, o CFO (Diretor Financeiro) da empresa Rossmann, numa reunião com todos os gerentes de loja, solicitou a cada um deles uma previsão diária de vendas, das próximas 6 semanas. Para um melhor resultado, essa demanda foi repassada para o time de Data Science.

# 3. Planejamento da Solução

## 3.1. Resumo
- O objetivo do projeto é fornecer previsões de vendas de 6 semanas, para várias lojas com base em dados históricos;
- Os potenciais métodos a serem utilizados são as séries temporais e regressão com algumas modificações;
- O formato de entrega será um Telegram Bot: ao digitar o ID da loja, o valor total de vendas será apresentado.

## 3.2. Metodologia

Neste projeto, a metodologia CRISP-DM foi implementada para entregar valor ao negócio. Em síntese, a estrutura do projeto Rossmann Sales Forecast foi a seguinte:

### 3.2.1. Entendimento do Negócio + Premissas + Limpeza de Dados:

- Os registros de vendas se iniciam em 1º de janeiro de 2013 e terminam em 31 de julho de 2015;
- Valores faltantes na variável `competition_distance` foram interpretados como ausência de competidores próximos ou lojas significativamente distantes que por isso, não são consideradas concorrentes. Logo, os valores NA foram alterados para 200.000 metros;
- Valores faltantes nas variáveis `competition_open_since_month` e `competition_open_since_year` podem decorrer da falta de competidores próximos ou do desconhecimento da data de abertura da loja concorrente. Dado isso, foi considerado o mês e ano do registro de vendas `date`;
- Valores faltantes nas variáveis na `promo_month_week` e `promo_month_year` também foram substituídos pela data do registro de vendas `date`;

**Após análise de dados, o conjunto final das variáveis ficou assim:**

colunas | descrição
------- | ---------
store | ID único para cada loja
day_of_week | dia da semana (1 = Segunda, 7 = Domingo)
date | data do registro de vendas
**sales** | **volume de vendas realizadas no dia do registro** 
customers | número de clientes que frequentaram a loja no dia do registro 
open | indica se a loja estava aberta (0 = fechado, 1 = aberto)
promo_week | indica se no registro de venda, a loja estava realizando uma promoção semanal
state_holiday | indica se a venda foi realizada num feriado estadual (a = feriado, b = feriado da Páscoa, c = Natal, 0 = nenhum)
school_holiday | indica se a venda no dia foi afetada pelo fechamento das escolas públicas
store_type | indica os 4 tipos de lojas que a a Rossmann possui (a, b, c, d)
assortment | descreve qual nível de sortimento de produtos a loja pertence (a = básico, b = extra, c = estendido)
competition_distance | distância em metros até a loja concorrente mais próxima
competition_open_since_month | indica o mês aproximado em que o concorrente mais próximo foi aberto
competition_open_since_year | indica o ano aproximado em que o concorrente mais próximo foi aberto
promo_month | indica se a loja participa de promoções mensais (0 = não participa, 1 = participa)
promo_month_since_week | descreve a semana em que a loja começou a participar das promoções mensais
promo_month_since_year | descreve o ano em que a loja começou a participar das promoções mensais
promo_interval | descreve os intervalos consecutivos em que promo_month é iniciado (as promoções mensais são realizadas 4 vezes no ano)
promo_month_active | indica se no dia do registro de venda, a loja estava participando das promoções do mês (0 = não estava, 1 = estava)
year | ano extraído da coluna data
month | mês extraído da coluna data
day | dia extraído da coluna data
week_of_year | semana do ano extraído da coluna data
competition_since | ano e mês extraído da coluna competition_open_since_year e competition_open_since_month
competition_time_month | indica há quantos meses a loja competidora existe
promo_month_since_date | ano e semana que começaram as promoções mensais
promo_month_time_week | quantidade de semanas que a loja participa das promoções mensais


### 3.2.2. Hipóteses + Exploração dos Dados

 - Visando organizar as ideias e listar características que podem ter impactado o faturamento de vendas das lojas Rossmann, o Mapa Mental de Hipóteses foi desenhado, e com o auxílio dele, 9 hipóteses foram criadas e testadas ao longo da fase de EDA.

<img src="https://imgur.com/q8Ot9z7.png" width="1200">

Após passar pelas análises (univariada, bivariada e multivariada), os principais insights obtidos através dos dados foram:

***H4. Lojas que participam das duas promoções deveriam vender mais***

**Hipótese Falsa** - Lojas que participam das duas promoções (semanal e mensal) vendem menos (e têm menor tempo de duração). Os registros de vendas mostram que as lojas vendem mais, em dias regulares e dias com promoção semanal. Há um certa complexidade quanto a essa variável, pois, existem características que podem impactar as vendas: o tipo de loja e o seu nível de sortimento, as necessidades dos clientes que podem não estar sendo atingidas, etc..
<img src="https://imgur.com/FqWz9TO.png" width="1200">

***H6. Lojas deveriam vender mais ao longo dos anos***

**Hipótese Falsa** - Lojas vendem menos ao longo dos anos. Analisando os dois anos completos (2013 e 2014) há uma queda nas vendas. Mas, pra não isolar o ano de 2015 (os registros de vendas vão até julho) foi analisado o comportamento (a proṕosito, crescente) do primeiro semestre dos três anos.
<img src="https://imgur.com/85tUf3J.png" width="1200">

### 3.2.3. Pré-processamento de dados e Seleção de Recursos
Chegou a etapa de preparação dos dados para que posteriormente ocorra a aplicação de algoritmos de aprendizagem de máquina. Como o aprendizado da maioria dos algoritmos de ML é facilitado com dados numéricos, técnicas foram necessárias: 
- ***de rescaling:*** Robust Scaler para as variáveis `competition_distance` e `competition_time_month` e Min-Max Scaler para `year` e `promo_month_time_week`;
- ***transformação - enconding:*** para variável `state_holiday`, foi utilizado One Hot Enconding; para a variável `store_type`, a Label Enconding; e para a variável `assortment`, a Ordinal Enconding;
- ***transformação - grandeza*:** na variável resposta `sales` foi usado o log1p, visando uma distribuição mais próxima do normal;
- ***transformação - natureza:*** nas variáveis temporais, o modelo de natureza cíclica foi utilizado, com o círculo trigonomêtrico (seno e coseno).

Com o Boruta, foi descoberto quais são as variáveis mais relevantes para usarmos no modelo preditivo de ML.

### 3.2.4. Algoritmos de Machine Learning

Foram testados 5 modelos diferentes (Average Model, Linear Regression Model, Lasso Regression Model, Random Forest Regressor e XGBoost Regressor).

Para avaliar o desempenho dos modelos, já que o pedido é uma previsão das próximas 6 semanas, para validar o modelo de ML, as últimas seis semanas de registros de vendas foram reservadas para validação e o restante do dataset, para treino.

**Avaliação da performance após o Cross-Validation**
Model Name | MAE CV | MAPE CV | RMSE
---------- | --------- | --------- | ---------
Random Forest Regressor | 696.66 | 0.1 | 1003.81 
XGBoost Regressor | 987.94 | 0.13 | 1407.43 
Linear Regression | 1980.04| 0.28 | 2840.0
Lasso Regression | 2014.53 | 0.28 | 2918.82

Ainda que o XGBoost não tenha obtido a melhor performance, o algoritmo foi escolhido para realizar a previsão de vendas, pois treina os dados com mais rapidez.

**O resultado final abaixo, com o ajuste de hiperparâmetros**

Model Name | MAE | MAPE | RMSE
---------- | --------- | --------- | ---------
XGBoost Regressor | 614.31 | 0.089 | 898.91 |


### 3.2.5. Avaliação do Desempenho do Algoritmo

O modelo é analisado sob uma perspectiva de negócios. As imagens abaixo mostram uma comparação de resultados, com o melhor e pior cenário de previsão. 

***Lojas com melhor desempenho - previsões com até 4% de erro***
<img src="https://imgur.com/UHiOq9r.png" width="1200">


***Lojas com pior desempenho - previsões com até 55% de erro***
<img src="https://imgur.com/JDyCY5l.png" width="1200">

Em vista disso, o total de vendas previstas para as próximas seis semanas se encontra no quadro abaixo:

Scenario	| Revenues (6w)
-------- | -------
predictions	| 285,743,680.00
worst_scenario	| 285,054,422.60
best_scenario	| 286,432,960.15

**Os gráficos abaixo mostram a performance do modelo.** 
- No gráfico ***Sales Variance***, as vendas reais e estimadas são comparadas;
- O gráfico ***%*** exibe a taxa de erro nas últimas 6 semanas. Uma taxa acima de 1, as predições estão superestimadas, enquanto menor que 1, estão subestimadas;
- O 3º gráfico ***Error Distribution***, distribuição do erro, é usado na análise do resíduo, uma teoria que o erro é analisado. O ideal é ter uma distribuição em forma de sino e com média zero;
- O último gráfico ***Residues***, exibe as previsões em relação ao erro. Espera-se que os erros estejam concentrados num tubo (erros com poucas variações).

<img src="https://imgur.com/bRHSw9P.png" width="1200">

### 3.2.6. Modelo em Produção

Para acesso rápido as previsões de vendas, foi criado um [Bot no Telegram](https://t.me/rssmnn_bot), usando Flask e Render. Como mostra o vídeo abaixo, é so enviar o número da loja que deseja, que uma mensagem será retornada com a previsão de vendas.

https://github.com/ctosta/Rossmann-Sales-Forecast/assets/84297748/20f61fd7-a515-49a5-91f2-aeb270bfb895


# 4.0. Conclusão e Próximos Passos

Para o primeiro ciclo do CRISP-DM, o modelo apresentou resultados medianos. É necessário que o projeto passe por outros ciclos a fim de melhorar a assertividade das previsões. Nos próximos ciclos podemos: 
- Adicionar mais variáveis;
- Criar novas hipóteses e validá-las;
- Analisar os erros acima de R$ 10.000 e criar modelos específicos para as lojas mais difíceis (MAPE alto).

