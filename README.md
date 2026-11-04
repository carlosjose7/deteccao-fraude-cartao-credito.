# Detecção de Fraude em Transações de Cartão de Crédito

## Problema

Este projeto tem como objetivo treinar modelos de Machine Learning para detectar transações fraudulentas em cartão de crédito. 

O grande desafio deste problema é o **desbalanceamento extremo**: 

- ~99,83% das transações são legítimas (Classe 0)
- ~0,17% das transações são fraudulentas (Classe 1)

### Por que o desbalanceamento muda a forma de avaliar?

Em datasets altamente desbalanceados, **acurácia é enganosa**. Um classificador que simplesmente prevê "não fraude" para todas as transações teria acurácia próxima a 99,83%, mas seria completamente inútil para o objetivo de negócio (não detectaria nenhuma fraude).

Por isso, neste projeto priorizamos métricas que realmente importam para detecção de fraude:

| Métrica | O que mede | Por que importa? |
| --- | --- | --- |
| **Recall (Sensibilidade)** | Dos casos reais de fraude, quantos o modelo conseguiu detectar? | É a métrica mais crítica. Um falso negativo (não detectar uma fraude) gera prejuízo direto. |
| **Precisão** | Das transações que o modelo marcou como fraude, quantas realmente são fraude? | Importante para evitar bloquear transações legítimas e gerar insatisfação nos clientes. |
| **F1-Score** | Média harmônica entre precisão e recall | Ótima para avaliar o equilíbrio entre detectar fraudes e minimizar falsos positivos em cenários desbalanceados. |
| **AUC-PR (Precision-Recall)** | Área sob a curva Precisão-Recall | Mais informativa que ROC-AUC quando há forte desbalanceamento. |

## Dataset

O dataset contém transações realizadas com cartões de crédito europeus em setembro de 2013. 

**Características:**
- `Time`: Segundos decorridos entre cada transação e a primeira transação no dataset
- `V1` a `V28`: Componentes principais obtidas por PCA (por questões de privacidade)
- `Amount`: Valor da transação
- `Class`: Variável alvo (1 = Fraude, 0 = Não Fraude)

**Fonte:** [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

O dataset é carregado diretamente pelo código via URL, não sendo versionado neste repositório.

## Preparação dos Dados

As seguintes etapas foram realizadas:

1. **Exploração**: Análise da distribuição das classes e estatísticas descritivas
2. **Feature Engineering**: Criação da variável `Log_Amount` = log(1 + Amount) para normalizar a distribuição assimétrica do valor das transações
3. **Padronização**: Aplicação do `StandardScaler` nas features `Time` e `Amount`/`Log_Amount`
4. **Divisão Treino/Teste**: Separação com `stratify=y` para manter a proporção exata de fraudes em ambos os conjuntos
5. **Tratamento do Desbalanceamento**: Uso de `class_weight` nos modelos e análise com diferentes abordagens

## Modelos Comparados

Foram avaliados os seguintes modelos:

| Modelo | Abordagem | Justificativa |
| --- | --- | --- |
| **Regressão Logística** | Baseline com `class_weight='balanced'` | Simples, interpretável e serve como ponto de referência |
| **Random Forest** | Ensemble com `class_weight='balanced_subsample'` | Robusto, lida bem com não-linearidades e fornece feature importance |
| **XGBoost** | Gradient Boosting com `scale_pos_weight` | Geralmente performa muito bem em detecção de fraudes |

## Limiar de Decisão e Explicabilidade

- **Limiar de Decisão**: Ao invés de usar o limiar padrão (0.5), foi analisado o trade-off entre Precisão e Recall através da curva Precision-Recall, ajustando o limiar para maximizar o Recall da classe fraude.
- **SHAP (SHapley Additive exPlanations)**: Utilizado para explicar global e localmente as predições do melhor modelo, identificando quais features mais contribuíram para classificar uma transação como fraude.

## Diferenças em Relação ao que foi Apresentado na Aula

- **Análise comparativa sistemática**: Comparação explícita entre os modelos com foco nas métricas corretas para dados desbalanceados
- **Ajuste de limiar**: Implementação prática do ajuste de threshold com base na curva Precision-Recall
- **Explicabilidade com SHAP**: Inclusão de análise SHAP para entender decisões individuais e globais do modelo
- **Foco em métricas corretas**: Ênfase deliberada em Recall/Precisão/F1 ao invés de acurácia, com justificação documentada

## Como Executar

1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Abra e execute o notebook `fraude_deteccao.ipynb` do início ao fim

**Nota:** O dataset será baixado automaticamente pela célula de carregamento.

## Tecnologias Utilizadas

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib, Seaborn
- SHAP
