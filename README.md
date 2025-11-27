# Sane.AI via Edge AI 💧🔍
## 1. Racional e Definição do Projeto
Título do Projeto: Sane.AI
Seleção da Trilha: Edge AI Application Track (Foco na implementação prática, otimização de código e inferência em hardware de baixo poder computacional).
Descrição (Abstract): Este projeto propõe o desenvolvimento de um Sistema de Aprendizado de Máquina em Borda (Edge ML) otimizado para a detecção anômala de vazamentos em infraestruturas hidráulicas pressurizadas urbanas. A metodologia baseia-se na aquisição e análise espectral de assinaturas vibracionais do subsolo. Em contraste com os sistemas acústicos tradicionais, que dependem exclusivamente de limiares de amplitude (volume) ou da intervenção humana especializada, o modelo emprega uma arquitetura de Deep Learning para discernir a característica espectral de baixa frequência ("o ronco da terra") de um vazamento persistente contra ruídos urbanos complexos e transientes (tráfego, operações industriais, pedestres). O processamento da inferência é realizado integralmente e em tempo real em um microcontrolador de baixo custo e restrição de memória.
Caso de Uso e Contexto de Mercado: A ineficiência hídrica é um desastre econômico e social no Brasil. Segundo o Estudo de Perdas de Água 2025 (Trata Brasil/GO Associados), o país desperdiça 40,31% de toda a água potável produzida nos sistemas de distribuição.
O Problema: Diariamente, o Brasil joga fora o equivalente a 6.346 piscinas olímpicas de água tratada. Cerca de 60% desse volume corresponde a perdas físicas (vazamentos na rede) , que poderiam abastecer 50 milhões de brasileiros anualmente se recuperadas.
A Falha Atual: A detecção desses vazamentos em cenários urbanos ruidosos é imprecisa. Métodos acústicos tradicionais geram falsos positivos e equipamentos de ponta são financeiramente inviáveis para monitorar a extensa malha de cidades com restrições orçamentárias.
A Solução Sane.AI: Um dispositivo de Borda (Edge Device) de baixo custo que utiliza Deep Learning para "ouvir" o vazamento real em meio ao caos urbano, atacando a maior fatia do desperdício nacional.
Justificativa Baseada em Dados (O Cenário Brasileiro): A relevância do Sane.AI é corroborada pelos dados do Estudo de Perdas de Água 2025, que evidenciam três pilares críticos para a implementação de hardware focado em perdas físicas:


### 1. A Predominância das Perdas Físicas (O Alvo do Projeto):
Ao contrário do senso comum de que as perdas se devem majoritariamente a fraudes, o estudo aponta que 60% do volume de água não faturada no Brasil corresponde a Perdas Físicas (Reais), ou seja, vazamentos na infraestrutura.

Implicação: O Sane.AI, ao utilizar análise vibracional, ataca a causa raiz da maior parte do desperdício, estimada em mais de 3 bilhões de m³ anuais.
### 2. Heterogeneidade Regional e o Paradoxo dos Grandes Centros: 
O Sane.AI é vital tanto para regiões com infraestrutura crítica quanto para grandes polos econômicos ineficientes.
Mercados Críticos (Norte/Nordeste): A solução é desenhada para escalar em regiões onde a infraestrutura é precária, como o Norte (49,78% de perdas) e Nordeste (46,25%). Casos extremos incluem Maceió (AL), que perde 71,73% de sua água , e Macapá (AP), com perdas superiores a 1.000 litros por ligação/dia.
O Caso do Rio de Janeiro: O estado do RJ apresenta perda na distribuição de 52,23%. Sua capital lidera o ranking negativo entre as capitais, desperdiçando 1.292,59 litros por ligação/dia, evidenciando a necessidade de filtragem de ruído avançada em metrópoles densas.
Volume em São Paulo: Mesmo em estados mais eficientes como São Paulo (32,66% de perdas), a densidade da malha exige tecnologia de precisão para reduzir o volume absoluto de desperdício.
### 3. Impacto Econômico e Monetização (O Business Case): 
Financeiramente, a ineficiência drena recursos bilionários. O custo total anual com perdas supera R$ 13 bilhões.
Custo Direto de Produção (OPEX): As Perdas Físicas, foco exclusivo do Sane.AI, representam um custo de produção "jogado fora" de aproximadamente R$ 2,4    bilhões ao ano.
Retorno: Cada vazamento detectado precocemente pelo dispositivo economiza diretamente R$ 0,79/m³ na conta de energia e insumos químicos da operadora.
Redução do Custo de Detecção (A Lógica Econômica): Conforme o conceito de "Nível Econômico de Vazamento", a viabilidade de reparar um vazamento depende do custo para detectá-lo.
Inovação: Atualmente, a detecção depende de equipamentos caros ou varredura humana lenta. Ao implementar um nó de borda (Edge Node) com microcontroladores acessíveis (Teensy), o Sane.AI reduz drasticamente o custo marginal de detecção, tornando economicamente viável a localização de micro vazamentos que hoje são ignorados por serem "caros demais" para encontrar.
Racional (Justificativa Técnica): A escolha pela classificação de áudio baseada em Redes Neurais Convolucionais 1D (1D-CNN) foi motivada pela falha demonstrada na análise univariada de amplitude para distinguir eventos transitórios (picos de volume) de vazamentos genuínos e persistentes. A plataforma Edge Impulse foi selecionada como ferramenta de MLOps (Machine Learning Operations) em Borda, viabilizando a otimização e a quantização do modelo para restrições estritas de hardware (ex: consumo de potência ultra baixo e memória RAM ≤ 100 KB).
.
2. Detalhamento do Processo e Iteração (A Jornada de Desenvolvimento)
O processo de engenharia e otimização do modelo (M.O.) foi iterativo, passando por quatro refatorações críticas baseadas na análise de métricas de desempenho e viés de dados (data bias).

Fase 1: O Viés de Dados "Limpos" e a Ilusão da Acurácia
Inicialmente, o treinamento foi realizado utilizando bibliotecas de áudio digitais e amostras coletadas na web. Estes dados representavam um cenário idealizado: áudios cristalinos, padronizados e sem interferências externas.
Falha: O modelo atingiu alta acurácia (85-95%) no painel de controle, mas a Matriz de Confusão revelou a falha crítica. O modelo sofreu de Overfitting aos dados limpos da web, tornando-se "surdo" para vazamentos reais em campo (baixo Recall) e incapaz de lidar com a complexidade acústica real.
Contexto Crítico: A Expedição ao Rio de Janeiro e a Física
Para corrigir a falta de sensibilidade gerada pelos dados artificiais, a equipe realizou uma expedição técnica intensiva de uma semana no Estado do Rio de Janeiro. O objetivo foi capturar a realidade acústica "suja" e não padronizada das tubulações urbanas.
Utilizando um Geofone de alta sensibilidade, a equipe percorreu diversas localidades estratégicas — Xerém, Ilha do Governador, a capital Rio de Janeiro e a Lapa — coletando um dataset robusto de vazamentos reais e cenários de não-vazamento (ruídos ambientes).
O contraste com os dados de laboratório foi imediato. O ambiente urbano caótico (trânsito pesado, obras, interferências) revelou o erro fundamental na abordagem inicial:
O Filtro Físico: Descobriu-se que o solo e o acoplamento do Geofone atuavam como um filtro Low Pass natural, atenuando drasticamente frequências acima de 1300 Hz. O "chiado" agudo presente nos dados da web não chegava ao sensor no mundo real.
Novo Foco de Frequência: A análise dos dados reais coletados nessa semana revelou que a assinatura única do vazamento que sobrevivia ao caos urbano residia nas baixíssimas frequências, especificamente entre 20 Hz e 40 Hz ("o ronco da terra").


Fase 2: A Barreira da Memória e a Solução 1D
Tentativas iniciais com Redes Neurais 2D (Visão Computacional) falharam devido à exaustão de memória (Failed to allocate bytes) no microcontrolador.
Solução Arquitetural: Migramos para uma 1D-CNN (Convolução Unidimensional).
O "Pulo do Gato" (Kernel Size): Para compensar a simplicidade da rede e permitir que ela enxergasse as ondas graves e lentas (20-40 Hz), aumentamos o Kernel Size para 7. Isso funcionou como uma "lente grande angular" temporal, elevando a detecção para 87%.

Fase 3: A Aposta no Espectrograma e o Limite de Performance
Guiados pelas análises iniciais do EON Tuner, implementamos uma arquitetura baseada em Espectrogramas de Alta Resolução. A hipótese era que a "visão computacional" aplicada ao som revelaria a textura sutil do vazamento.
O Teste: Configuramos o DSP com FFT de 512 e treinamos uma CNN para analisar as imagens espectrais.
O Resultado: O modelo atingiu um platô de ~86% de acurácia. Embora fosse um bom resultado, ainda havia uma taxa residual de confusão entre ruídos complexos e vazamentos.
O Gargalo: Percebemos que, para aumentar a precisão, precisávamos de mais contexto temporal (analisar janelas de tempo maiores), mas o Espectrograma era "pesado" demais computacionalmente. Aumentar o tempo com essa técnica estouraria a memória do microcontrolador. Estávamos travados.

Fase 4: O Pivô Estratégico (Tempo > Resolução Visual)
Decidimos mudar a abordagem. A diferença entre um carro passando e um vazamento não estava apenas no detalhe da frequência, mas na persistência do som ao longo do tempo.
A Mudança Radical: Aumentamos a janela de amostragem de 2 segundos para 5 segundos (5000 ms). Isso permitiu que a IA "ouvisse" a história completa do som.
O Retorno ao MFE: Para processar 5 segundos de áudio sem travar o hardware, substituímos o Espectrograma pesado pelo MFE (Mel-Filterbank Energy).
O Ganho: O MFE comprime a informação de frequência de forma eficiente. Ao combiná-lo com a janela de 5 segundos, conseguimos processar um contexto temporal 2,5x maior.
Resultado: Essa alteração simples, mas estratégica, nos deu o ganho crítico de +2% de acurácia imediata e, mais importante, eliminou a instabilidade nas detecções.
Fase 5: A Solução Final Híbrida (Feature Fusion & Dense Network)
Com a entrada de dados otimizada (Janela de 5s + MFE), refinamos o "cérebro" do sistema para a arquitetura final de implantação. Em vez de confiar apenas no MFE, criamos uma arquitetura de Fusão de Características (Feature Fusion):
Entrada Combinada: O modelo recebe simultaneamente os dados do MFE (a assinatura auditiva humana) e as Spectral Features (estatística matemática bruta do sinal).
Arquitetura Neural: Substituímos a CNN por uma Rede Neural Densa Profunda (Deep Dense Network) com camadas de 128 e 32 neurônios e Dropout de 0.25.
Por que Dense Network? Como o MFE e as Spectral Features já entregam os dados "mastigados" e resumidos, uma rede densa é mais rápida e eficiente para tomar a decisão final do que uma convolucional.
Validação Final: Esta combinação atingiu o "Santo Graal" da detecção de vazamentos: 100% de Recall (Vazamentos Detectados) e uma robustez incomparável contra falsos positivos no cenário urbano real.

3. Qualidade e Uso do Conjunto de Dados
Documentação do Dataset 
Os dados foram coletados em cenários de campo reais, utilizando um transdutor sísmico de banda larga (Geofone) com acoplamento mecânico adaptado para a captação de vibrações de solo.
Classe LEAK: Gravações de vazamentos reais em diversas pressões e materiais de tubulação.
Classe NO_LEAK: Conjunto robusto de gravações de ruídos ambientais urbanos, abrangendo tráfego veicular, vibrações estruturais, ruídos impulsivos (passos, vozes) e o silêncio operacional do sistema.
Representatividade e Ajustes 
A representatividade foi assegurada pela diversidade da classe NO_LEAK, com foco em mitigar o risco de data bias. Foi realizada uma coleta proposital de áudios contendo ruídos impulsivos de alta amplitude (ex: motocicletas passando) para treinar explicitamente a IA a distinguir que a Alta Amplitude é ortogonal à classificação de Vazamento, forçando-a a focar na assinatura de frequência persistente.
Abertura e Licença: O dataset foi curado para anonimização de informações sensíveis (remoção de vozes identificáveis) e será disponibilizado sob licença permissiva (MIT/Apache) para validação e reprodução.
Link para o Dataset: https://studio.edgeimpulse.com/public/833695

4. Componentes de Hardware e Reprodutibilidade
Hardware Utilizado 
Sensor de Aquisição: Geofone (transdutor sísmico de baixa frequência).
Microcontrolador (Host): Teensy 4.1 (Microcontrolador de alta performance baseado no core ARM Cortex-M7 a 600 MHz).
Interface: Conversor Analógico-Digital (ADC) Integrado de alta resolução e baixa latência.
Diagrama de Fluxo (Pipeline Híbrido de Edge ML)
        O fluxo de dados foi desenhado para processamento paralelo no microcontrolador:

Aquisição: Captura de vibração a 48 kHz em janelas de 5 segundos.
Pré-processamento Paralelo (Dual DSP):
Via A: Geração de Espectrograma (FFT 128) para análise visual.
Via B: Extração de Características Espectrais (Spectral Features) para análise estatística.
Inferência Simultânea:
A CNN processa o Espectrograma buscando padrões de textura.
A Rede Densa processa as estatísticas buscando anomalias de energia.
Fusão de Decisão: O firmware avalia as saídas dos dois modelos. A classificação final de "Vazamento" é resultado da combinação inteligente dessas duas inteligências distintas.
Código e Repositório 
Todo o código fonte da aplicação, incluindo a biblioteca otimizada exportada do Edge Impulse e a lógica de aplicação de Pós-processamento em C++ (main.cpp), está versionado:
GitHub: https://github.com/Sanesoluti-dev/Cod_teensy
Protocolo de Reprodução Deste Projeto
Clone o Projeto: Acessar e clonar o projeto público na plataforma Edge Impulse: https://studio.edgeimpulse.com/studio/833889
Exportação do Modelo: Exportar o modelo treinado como uma biblioteca C++ otimizada para Edge.
Compilação: Utilizar o código-fonte de aplicação fornecido no repositório GitHub para compilar o firmware para o hardware alvo (Teensy 4.1).
Teste de Validação: Utilizar um gerador de ruído branco de baixa frequência para simular a assinatura de vazamento e inputs impulsivos (ex: toque ou batida no sensor) para simular eventos de "Não Vazamento", validando a robustez do classificador.
5. Conclusão do Projeto
O Sane.AI inova ao trazer para a borda (Edge) uma complexidade geralmente reservada à nuvem: a inferência multi-modelo.
Nossa conclusão é que a robustez necessária para o ambiente urbano caótico não vem de um único "algoritmo mágico", mas da orquestração de diferentes técnicas. Ao combinar a visão computacional (CNN sobre Espectrograma) com a análise estatística espectral na mesma janela de 5 segundos, criamos um dispositivo que possui, efetivamente, "dois cérebros".
O resultado é um sensor que não apenas escuta, mas valida o que ouviu, garantindo a precisão necessária para combater o desperdício de água no planeta.
6. Referências de Pesquisa e Fontes
[1] Ministério das Cidades and Secretaria Nacional de Saneamento, "Relatório SINISA: Diagnóstico dos Serviços de Água e Esgotos 2024 (Ano-base 2023)," Brasília, Brazil, 2024. [Online]. Available: https://www.gov.br/cidades/pt-br/acesso-a-informacao/acoes-e-programas/saneamento/sinisa/resultados-sinisa/copy_of_RELATORIO_SINISA_ABASTECIMENTO_DE_AGUA_2024.pdf
[2] Instituto Trata Brasil and GO Associados, "Estudo de Perdas de Água 2025 (SNIS 2023 base year): Desafios para a Universalização do Saneamento," São Paulo, Brazil, 2025. [Online]. Available: https://tratabrasil.org.br/
[3] Edge Impulse Inc., "Audio Classification with MFE and Spectral Features on Edge Devices," Edge Impulse Documentation, 2024. [Online]. Available: https://docs.edgeimpulse.com/docs/tutorials/audio-classification
[4] PJRC, "Teensy 4.1 Development Board Specifications (ARM Cortex-M7 at 600 MHz)," 2024. [Online]. Available: https://www.pjrc.com/store/teensy41.html


        	

