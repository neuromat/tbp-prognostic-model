Aqui encontram-se os resultados gerados para a tese de doutorado de Cristiane Borges Patroclo, do Programa de Pós-Graduação em Fisiologia, do Instituto de Biofísica Carlos Chagas Filho da Universidade Federal do Rio de Janeiro (UFRJ). O trabalho buscou identificar em pacientes adultos os preditores prognósticos das lesões traumáticas do plexo braquial (LTPB) por meio de mineração de dados e florestas aleatórias (FA).

Os resultados incluem:
- os dados dos pacientes com LTPB (em formato .CSV), (provenientes de respostas a questionários eletrônicos)
- os dados (em formato .CSV) gerados pelo pré-processamento, usados no treinamento dos modelos preditivos
- os modelos de FA treinados (em formato .Pickle)
- os gráficos (em .PNG e .SVG) de importância de atributos e de contribuição de valores de atributos
- os dados usados para a geração dos gráficos
- relatórios de desempenho dos modelos

Os resultados estão separados em seis diferentes grupos, de acordo com o momento (em relação à data da lesão) do desfecho da recuperação considerado no treinamento do modelo:
- grupo 1: desfecho após 1,5 anos (desempate pelo mais próximo de 1,5 anos)
- grupo 2: desfecho após 1,5 anos (desempate pelo mais distante de 1,5 anos)
- grupo 3: desfecho após 3 anos (desempate pelo mais próximo de 3 anos)
- grupo 4: desfecho após 3 anos (desempate pelo mais distante de 3 anos)
- grupo 5: desfecho após 2 anos (desempate pelo mais próximo de 2 anos)
- grupo 6: desfecho após 2 anos (desempate pelo mais distante de 2 anos)

Em cada um dos grupos, os resultados estão organizados em 3 pastas:
- "into", com os resultados obtidos para os pacientes do Instituto Nacional de Traumatologia e Ortopedia (INTO)
- "indc", com os resultados obtidos para os pacientes do Instituto de Neurologia Deolindo Couto (INDC)
- "all",  com os resultados obtidos para os pacientes do INTO e INDC juntos

Além disso, dentro de cada uma das pastas citadas acima, os resultados ficar divididos por tipo de alvo de recuperação de LTPB. São quatros os alvos:
- "lstPexMuscstrength_ElbowFlex" - flexão do cotovelo
- "lstPexMuscstrength_ShoulderAbduc" - abdução do ombro
- "lstPexMuscstrength_ShoulderExrot" - rotação externa do ombro
- "yonPexPain" - dor
