# Problema da Mochila 0/1 resolvido com Algoritmo Genético

- **Nome completo:** Talita Marina Viana Luz

---

## Introdução

### Contexto

A Otimização Combinatória estuda problemas em que se busca a melhor solução
(ótima ou próxima do ótimo) dentro de um conjunto finito, porém geralmente
muito grande, de soluções possíveis. Um dos problemas clássicos dessa área é
o **Problema da Mochila 0/1 (0/1 Knapsack Problem)**.

Dado um conjunto de `n` itens, cada um com um **peso** e um **valor**, e uma
mochila com **capacidade máxima** `W`, o objetivo é escolher um subconjunto
de itens que **maximize o valor total** carregado, sem que a **soma dos
pesos ultrapasse a capacidade** da mochila. Cada item só pode ser incluído
inteiramente (1) ou não incluído (0) — não é permitido fracionar itens, o
que diferencia esta versão da "mochila fracionária" (essa sim resolvida de
forma trivial por um algoritmo guloso).

O problema tem aplicações práticas em logística e carregamento de cargas,
alocação de orçamento/investimentos, corte de materiais, seleção de projetos
com recursos limitados, empacotamento de dados em mídias de armazenamento,
entre outros cenários em que recursos limitados precisam ser distribuídos
entre alternativas de custo/benefício distintos.

### Problema escolhido

- **Categoria:** Problema da Mochila 0/1 (0/1 Knapsack Problem).

### Estratégia algorítmica escolhida

Foram implementadas **duas** estratégias, para permitir comparação:

1. **Método exato — Programação Dinâmica (DP):** usado como "gabarito", ou
   seja, para calcular a solução ótima real do problema e assim poder medir
   o quão boa é a solução heurística.
2. **Metaheurística — Algoritmo Genético (AG):** estratégia principal do
   trabalho, inspirada no processo de seleção natural (Darwin), utilizando
   uma população de soluções candidatas que evolui ao longo de gerações por
   meio de seleção, cruzamento (crossover) e mutação.

### Método exato ou heurística?

- A **Programação Dinâmica** é um **método exato**: sempre encontra a
  solução ótima, mas sua complexidade de tempo e espaço é **O(n · W)**
  (pseudo-polinomial). Isso significa que, embora seja "polinomial" em `n`
  e `W`, o valor de `W` pode ser exponencialmente grande em relação ao
  número de bits usados para representá-lo, tornando o método impraticável
  para capacidades muito grandes.
- O **Algoritmo Genético** é uma **metaheurística**: não garante encontrar a
  solução ótima, mas geralmente encontra soluções de boa qualidade em tempo
  computacional muito menor e de forma escalável, mesmo em instâncias
  grandes onde a Programação Dinâmica se tornaria inviável.

### Complexidades

| Método                 | Tipo         | Complexidade de tempo                         | Garante o ótimo? |
|-------------------------|--------------|------------------------------------------------|-------------------|
| Programação Dinâmica    | Exato        | O(n · W)                                        | Sim               |
| Algoritmo Genético      | Metaheurística | O(gerações × população × n)                   | Não               |

Onde `n` é o número de itens e `W` é a capacidade da mochila.

---

## Desenvolvimento

### Dados do problema

Como o enunciado não exigiu um dataset específico, foi implementado um
**gerador de instâncias sintéticas** (`generate_instance`), com semente
(`seed`) fixa para garantir reprodutibilidade. A instância usada nos testes
possui:

- **40 itens**, com pesos aleatórios entre 5 e 50 e valores aleatórios entre
  10 e 100;
- **Capacidade da mochila** igual a 50% da soma total dos pesos de todos os
  itens (uma mochila "moderadamente apertada").

A instância gerada é salva em `results/instance.json`, podendo ser
inspecionada ou substituída por outra instância (real ou de benchmark) sem
alterar o restante do código.

### Modelagem matemática

**Variáveis de decisão:**