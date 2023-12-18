Este repositório contém os códigos dos programas, dados e gráficos produzidos para o trabalho de doutorado intitulado "PREDITORES PROGNÓSTICOS EM PACIENTES ADULTOS COM LESÃO TRAUMÁTICA DE PLEXO BRAQUIAL", de Cristiane Borges Patroclo, do Programa de Pós-Graduação em Fisiologia, do Instituto de Biofísica
Carlos Chagas Filho da Universidade Federal do Rio de Janeiro (UFRJ). 
O trabalho buscou identificar em pacientes adultos os preditores prognósticos das lesões traumáticas do plexo braquial (LTPB) por meio de mineração de dados e florestas aleatórias (FA).

Os programas usados para a criação dos modelos preditivos baseados em FAs deste trabalho são adaptações do [código-fonte](https://github.com/luumelo14/prognostic-model) escrito por Luciana de Melo Abud em seu [mestrado em 2018](https://www.teses.usp.br/teses/disponiveis/45/45134/tde-20082018-140641/). 

As adaptações foram feitas pela Profa. Dra. Kelly Rosa Braghetto (DCC-IME-USP) e tiveram dois objetivos principais: (i) adequar o procedimento de pré-processamento de dados ao novos dados para treinamento dos modelos, produzidos no trabalho de Patroclo, e (ii) criar novos tipos de gráficos para a visualização da contribuição dos valores dos atributos selecionados para os modelos. O algoritmo de geração das FAs foi mantido como o original. 

Na pasta [prognostic-model](https://github.com/neuromat/tbp-prognostic-model/tree/master/cristiane-patroclo-thesis-results), encontram-se os resultados gerados para a tese de Cristiane Borges Patroclo:
- os dados dos pacientes com LTPB (no formato de respostas a questionários eletrônicos)
- os dados gerados pelo pré-processamento, usados no treinamento dos modelos preditivos
- os modelos treinados
- os gráficos de importância de atributos e de contribuição de valores de atributos
- os dados usados para a geração dos gráficos
- relatórios de desempenho dos modelos

Na pasta [src](https://github.com/neuromat/tbp-prognostic-model/tree/master/src), encontram-se os códigos dos programas usados para a geração dos resultados listados acima. 

