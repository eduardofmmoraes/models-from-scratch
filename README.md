# models-from-scratch

Implementações de modelos de Machine Learning **do zero, usando exclusivamente NumPy**.

O objetivo do projeto é entender a fundo como cada modelo funciona: forward pass, funções de perda, cálculo de gradientes (backpropagation) e otimização. Nenhum framework de ML (scikit-learn, PyTorch, TensorFlow, JAX…) é usado na implementação dos modelos. Toda a matemática é escrita à mão com operações vetorizadas do NumPy.

## Objetivo

- **Modelos 100% NumPy**: tudo o que está dentro de `src/models_from_scratch/` (camadas, ativações, perdas, otimizadores, métricas e o treino) depende apenas de `numpy`.
- **Bibliotecas auxiliares só nas bordas**: `pandas`, `scikit-learn`, `matplotlib` e `streamlit` aparecem só nos scripts de exemplo, para **carregar datasets**, pré-processar dados tabulares e visualizar resultados. Elas nunca participam do aprendizado.

## Modelos implementados

### Regressão Logística
[`logistic_regression/logistic_regressor.py`](src/models_from_scratch/logistic_regression/logistic_regressor.py)

- Classificação binária com função sigmoide
- Perda: Binary Cross-Entropy
- Otimização: Gradient Descent (batch completo)
- Exemplo: dataset **Adult** (OpenML), que prevê se a renda anual passa de 50K

### Rede Neural (MLP)
[`neural_network/`](src/models_from_scratch/neural_network/)

| Componente | Implementações |
|---|---|
| Camadas | `Dense` |
| Ativações | `ReLU`, `Softmax`, `Sigmoid`, `Linear` |
| Perdas | `CrossEntropy` (com gradiente combinado Softmax + Cross-Entropy) |
| Otimizadores | `SGD` com momentum e decaimento exponencial da learning rate |
| Métricas | `Accuracy_Category` |
| Treino | mini-batches, backpropagation, métricas acumuladas por época |

- Exemplo: dataset **Digits** (dígitos manuscritos 8x8), classificação em 10 classes

## Estrutura do projeto

```
models-from-scratch/
├── src/
│   ├── utils.py                      # utilitários (ex.: train_test_split em NumPy)
│   └── models_from_scratch/
│       ├── logistic_regression/
│       │   ├── logistic_regressor.py # modelo
│       │   └── main.py               # exemplo de treino (Adult)
│       └── neural_network/
│           ├── layers.py
│           ├── activation_functions.py
│           ├── loss.py
│           ├── optimizers.py
│           ├── accuracy.py
│           ├── network.py            # orquestra forward/backward/fit
│           └── main.py               # exemplo de treino (Digits)
├── pyproject.toml
└── uv.lock
```

## Instalação

O projeto usa [uv](https://docs.astral.sh/uv/) e requer Python >= 3.12.

```bash
git clone <url-do-repositorio>
cd models-from-scratch
uv sync
```

## Executando os exemplos

```bash
# Regressão logística no dataset Adult
uv run python -m models_from_scratch.logistic_regression.main

# Rede neural no dataset Digits
uv run python -m models_from_scratch.neural_network.main
```

> O dataset Adult é baixado do OpenML na primeira execução, então é preciso ter conexão com a internet.

## Exemplo de uso

```python
from models_from_scratch.neural_network.layers import Dense
from models_from_scratch.neural_network.activation_functions import Softmax
from models_from_scratch.neural_network.loss import CrossEntropy
from models_from_scratch.neural_network.optimizers import SGD
from models_from_scratch.neural_network.accuracy import Accuracy_Category
from models_from_scratch.neural_network.network import NeuralNetwork

nn = NeuralNetwork([
    Dense(64, 64),               # ReLU por padrão
    Dense(64, 64),
    Dense(64, 10, Softmax),
])
nn.set(
    optimizer=SGD(learning_rate=1e-2, momentum=0.9, decay_rate=1e-4),
    accuracy=Accuracy_Category(),
    loss=CrossEntropy(),
)

nn.fit((X_train, y_train), epochs=100, batch_size=32)
print(nn.predict(X_test, y_test))  # acurácia no conjunto de teste
```

## Roadmap

- [x] Regressão Logística
- [x] Rede Neural densa (MLP) com SGD + momentum
- [ ] Regressão Linear (equação normal e gradient descent)
- [x] Backward de `Sigmoid`/`Linear` e perdas para regressão (MSE, MAE)
- [ ] Otimizadores: Adam, RMSProp, AdaGrad
- [ ] Regularização (L1/L2) e Dropout
- [ ] K-Nearest Neighbors
- [ ] K-Means
- [ ] Naive Bayes
- [ ] Árvore de Decisão e Random Forest
- [ ] PCA
- [ ] SVM
- [ ] Testes comparando os resultados com implementações de referência

## Licença

Distribuído sob a licença MIT. Veja [LICENSE](LICENSE).
