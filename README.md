# Detecção de Prompt Injection em LLMs com Machine Learning clássico

Classificação de textos como **benignos** ou **ataques de prompt injection**, comparando **5 classificadores** (kNN, Árvore de Decisão, Perceptron, SVM com kernel RBF e Shrinkage Discriminant Analysis) sobre **2 representações do texto** (TF-IDF e embeddings do RoBERTa).

> **Pergunta central:** qual combinação de representação e classificador detecta melhor os ataques, sem recorrer a um segundo LLM?

## Contexto

*Prompt injection* é um ataque em que o texto de entrada tenta **sobrescrever as instruções originais** de um modelo de linguagem (por exemplo, "esqueça as regras anteriores e..."). Detectar esses textos antes que cheguem ao LLM é uma camada de defesa barata e rápida, e é o que este projeto investiga com modelos clássicos de aprendizado de máquina.

## Dados

| Item | Descrição |
|---|---|
| Tamanho | **5.000 exemplos** com colunas `text`, `label`, `source` e `id` |
| Classes | **1.021 ataques** (`label = 1`) e **3.979 benignos** (`label = 0`), aproximadamente 20% / 80% |
| Divisão | 80% treino / 20% teste (**1.000 exemplos, sendo 796 benignos e 204 ataques**), estratificada, `random_state = 42` |
| Fonte | Amostra `piguard_sample_5k.csv`, reunida a partir de **22 conjuntos de dados públicos** (coluna `source`), entre eles TaskTracker, safe-guard-prompt-injection, hackaprompt, BIPIA, Alpaca e ultrachat_200k. As licenças variam por fonte |

O desbalanceamento é moderado, então **não foi usado oversampling nem SMOTE**. Em vez disso, as classes recebem pesos (`class_weight='balanced'`, ou priors 0,8/0,2 no SDA).

## Representações do texto

Os vetores já vêm prontos na pasta `piguard_sample/embeddings/` (um `.npy` por representação, com 5.000 linhas na mesma ordem do CSV). Este projeto usa duas delas:

| Representação | Tratamento |
|---|---|
| **TF-IDF** (300 dimensões) | Valores já entre −0,70 e 0,75, então **não foi normalizado** |
| **RoBERTa** (768 dimensões) | Valores entre −1,35 e 12,54, então foi **normalizado**: `StandardScaler` (Perceptron e SDA) e `RobustScaler` (kNN e SVM) |

A pasta traz também embeddings de **fastText, GloVe, Word2Vec (300 dimensões), BERT e ALBERT (768 dimensões)**, que não foram usados aqui e ficam como extensão natural do trabalho.

Toda normalização é ajustada **só no treino**. No SDA com RoBERTa, ela fica dentro de um `Pipeline` para evitar vazamento de informação na validação cruzada.

## Modelos e hiperparâmetros

| Modelo | Configuração |
|---|---|
| **kNN** | `GridSearchCV` (5 folds, F1) sobre `n_neighbors` {3, 5, 7, 11}, `weights` {uniform, distance} e `metric` {cosine, euclidean} |
| **Árvore de Decisão** | `GridSearchCV` sobre `max_depth` {10, 20, None}, `min_samples_split` {2, 5} e `min_samples_leaf` {1, 2}, com `class_weight='balanced'` |
| **Perceptron** | Sem penalização, `max_iter = 1000`, `class_weight='balanced'` |
| **SVM** | Kernel RBF, `GridSearchCV` sobre `C` {0,1, 1, 10} e `gamma` {scale, auto}, com `class_weight='balanced'` |
| **SDA** (*Shrinkage Discriminant Analysis*) | `LinearDiscriminantAnalysis` com solver `lsqr` e priors 0,8/0,2. Dois estimadores de *shrinkage*: **Ledoit-Wolf** (`auto`) e **OAS** (λ = 0,0495 no TF-IDF e 0,0023 no RoBERTa) |

A métrica de seleção é o **F1-score**, adequada ao desbalanceamento. O conjunto de teste só é usado na avaliação final.

## Resultados

Desempenho no **conjunto de teste** (1.000 exemplos). Precisão, recall e F1 são da classe **Prompt Injection**.

### TF-IDF

| Modelo | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| kNN | 0,91 | 0,89 | 0,64 | 0,74 |
| Árvore de Decisão | 0,82 | 0,55 | 0,56 | 0,56 |
| Perceptron | 0,89 | 0,71 | 0,78 | 0,74 |
| SVM (RBF) | **0,92** | 0,84 | **0,77** | **0,80** |
| SDA (Ledoit-Wolf) | 0,91 | **0,94** | 0,58 | 0,72 |

### RoBERTa

| Modelo | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| kNN | 0,93 | 0,88 | 0,78 | 0,83 |
| Árvore de Decisão | 0,85 | 0,63 | 0,67 | 0,65 |
| Perceptron | 0,93 | 0,79 | **0,87** | 0,83 |
| **SVM (RBF)** | **0,95** | **0,93** | 0,83 | **0,88** |
| SDA (OAS) | 0,94 | 0,90 | 0,77 | 0,83 |

> Para o SDA, o notebook usa a variante de melhor desempenho em cada representação. No TF-IDF, o Ledoit-Wolf ficou levemente acima do OAS. No RoBERTa, as duas variantes empataram (F1 de 0,83), e o OAS foi mantido.

### Principais observações

- **Melhor combinação: SVM com kernel RBF + RoBERTa**, com acurácia de 0,954, precisão de 0,934, recall de 0,833 e F1 de 0,881 na classe de ataque.
- **Os embeddings do RoBERTa superaram o TF-IDF em todos os modelos**, com ganho de F1 de cerca de 0,08 (SVM, kNN e Perceptron), 0,09 (Árvore) e 0,11 (SDA). A representação contextual captura melhor a intenção do texto do que a frequência de termos.
- **O recall da classe de ataque é o ponto mais fraco**: mesmo no melhor modelo, cerca de 17% dos ataques não são detectados. Em segurança, isso importa mais do que a acurácia global.
- **O SDA no TF-IDF mostra o contraste entre precisão e recall**: quase não gera falsos positivos (precisão 0,94), mas deixa passar 42% dos ataques (recall 0,58).
- **A Árvore de Decisão foi o pior modelo nas duas representações**, o que é comum em vetores densos de alta dimensão, em que divisões por um único atributo capturam pouca informação.
- **O Perceptron**, mesmo sendo um modelo linear simples, teve o maior recall com RoBERTa (0,87), à custa de precisão menor (0,79).

## Limitações

- **Um único split** de treino e teste (semente 42), sem repetições. Diferenças de poucos pontos percentuais entre modelos podem estar dentro da variação amostral.
- **Só 204 ataques no teste**, então cada ataque perdido altera o recall em ~0,5 ponto.
- As curvas Precision-Recall e ROC são geradas no notebook, mas o desempenho **depende do limiar de decisão**, que ficou no padrão de cada modelo. Ajustá-lo para priorizar o recall seria um próximo passo natural.
- **A fonte do exemplo está muito ligada ao rótulo.** Sete fontes só têm ataques (448 exemplos, 44% dos ataques) e nove só têm exemplos benignos (2.703, 68% dos benignos). Por exemplo, os 317 exemplos do hackaprompt são todos ataques, e Alpaca, ultrachat_200k e open-instruct são todos benignos. Os modelos podem estar aprendendo o **estilo da fonte** além da intenção do ataque. Avaliar deixando uma fonte de fora (*leave-one-source-out*) mostraria quanto o resultado se mantém em dados de origem nova.
- A generalização para **ataques novos**, ou para outros idiomas e estilos, não foi avaliada.
- Os embeddings vêm pré-computados, e o código que os gerou não faz parte do projeto. Por isso, não é possível conferir como o TF-IDF foi ajustado (por exemplo, se antes da divisão treino/teste) nem reproduzir os vetores do zero.

## Como executar

```bash
git clone https://github.com/<seu-usuario>/<nome-do-repositorio>.git
cd <nome-do-repositorio>
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook Detection_of_Prompt_Injection_Attacks_in_LLMs.ipynb
```

O dataset e os embeddings já estão no repositório, na pasta `piguard_sample/`, e o notebook os lê dos caminhos locais:

```python
df = pd.read_csv("piguard_sample/piguard_sample_5k.csv", encoding="latin1")
tfidf   = np.load("piguard_sample/embeddings/tfidf.npy")
roberta = np.load("piguard_sample/embeddings/roberta.npy")
```

Também roda no **Google Colab**: basta enviar a pasta `piguard_sample/` ou clonar o repositório. Os resultados são reprodutíveis (semente 42).

## Estrutura

```
.
├── Detection_of_Prompt_Injection_Attacks_in_LLMs.ipynb
├── piguard_sample/
│   ├── piguard_sample_5k.csv          # 5.000 exemplos (text, label, source, id)
│   ├── load_embeddings.ipynb          # exemplo de carregamento (BERT + regressão logística)
│   └── embeddings/                    # tfidf, roberta (usados) + fasttext, glove, word2vec, bert, albert
└── README.md
```

## Tecnologias

Python · scikit-learn · NumPy · pandas · Matplotlib · RoBERTa (embeddings) · TF-IDF

## Referências

- [kNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) · [Árvore de Decisão](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) · [Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html) · [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) · [LDA / SDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

## Autoria

- `<Anysabele>`, `<Maria Clara>`, `<Millena>`
