"""
==============================================================================
 PROBLEMA DA MOCHILA 0/1 (0/1 KNAPSACK PROBLEM)
 Estratégia algorítmica: ALGORITMO GENÉTICO (metaheurística)
 Solver de referência: PROGRAMAÇÃO DINÂMICA (método exato) - usado apenas
                        para avaliar a qualidade da solução heurística.

 Disciplina: Otimização Combinatória
 Trabalho individual
==============================================================================

DESCRIÇÃO DO PROBLEMA
----------------------
Dado um conjunto de n itens, cada um com um peso (weight) e um valor (value),
e uma mochila com capacidade máxima W, o objetivo é selecionar um subconjunto
de itens que:
    - Maximize a soma dos valores dos itens selecionados;
    - Respeite a restrição de que a soma dos pesos não ultrapasse W.

Cada item só pode ser escolhido inteiramente (1) ou não escolhido (0) -
por isso "0/1" - não é permitido fracionar itens (o que caracterizaria a
versão "fracionária", resolvida de forma trivial e gulosa).

Este é um problema NP-difícil (NP-hard) no sentido forte quando os pesos e
capacidade são tratados como parte da entrada em codificação binária, mas
admite um algoritmo pseudo-polinomial exato via Programação Dinâmica,
com complexidade O(n*W). Para instâncias com W muito grande, esse método
exato se torna impraticável, o que justifica o uso de metaheurísticas
(como o Algoritmo Genético aqui implementado) para obter boas soluções
em tempo hábil, mesmo sem garantia de otimalidade.

FORMULAÇÃO MATEMÁTICA
----------------------
Variáveis de decisão:
    x_i ∈ {0, 1}, i = 1..n   (x_i = 1 se o item i é escolhido, 0 caso contrário)

Função objetivo (maximização):
    max  Σ (value_i * x_i),  i = 1..n

Restrição:
    Σ (weight_i * x_i) <= W,  i = 1..n

==============================================================================
"""

import random
import json
import csv
import time
from dataclasses import dataclass, field
from typing import List, Tuple

# Usamos matplotlib apenas para gerar o gráfico de convergência do GA.
import matplotlib
matplotlib.use("Agg")  # backend não interativo (não exige tela/gráfico)
import matplotlib.pyplot as plt


# ==============================================================================
# 1. MODELAGEM DO PROBLEMA
# ==============================================================================

@dataclass
class Item:
    """Representa um item da mochila: peso e valor."""
    id: int
    weight: int
    value: int


@dataclass
class KnapsackInstance:
    """Representa uma instância completa do problema da mochila 0/1."""
    items: List[Item]
    capacity: int

    @property
    def n(self) -> int:
        return len(self.items)


def generate_instance(n_items: int = 40, capacity_ratio: float = 0.5,
                       seed: int = 42) -> KnapsackInstance:
    """
    Gera (aleatoriamente, com semente fixa para reprodutibilidade) uma
    instância do problema da mochila.

    - n_items: número de itens disponíveis.
    - capacity_ratio: capacidade da mochila como fração da soma total
      dos pesos de todos os itens (controla o quão "apertada" é a mochila).
    - seed: semente do gerador aleatório, para que os resultados sejam
      reproduzíveis por qualquer pessoa que execute o código.
    """
    rng = random.Random(seed)
    items = [
        Item(id=i, weight=rng.randint(5, 50), value=rng.randint(10, 100))
        for i in range(n_items)
    ]
    total_weight = sum(it.weight for it in items)
    capacity = int(total_weight * capacity_ratio)
    return KnapsackInstance(items=items, capacity=capacity)


def save_instance_json(instance: KnapsackInstance, path: str) -> None:
    """Salva a instância gerada em um arquivo JSON, para documentação/reuso."""
    data = {
        "capacity": instance.capacity,
        "items": [{"id": it.id, "weight": it.weight, "value": it.value}
                   for it in instance.items],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ==============================================================================
# 2. MÉTODO EXATO: PROGRAMAÇÃO DINÂMICA (usado como "gabarito" / baseline)
# ==============================================================================

def solve_exact_dp(instance: KnapsackInstance) -> Tuple[int, List[int]]:
    """
    Resolve o problema da mochila 0/1 de forma ÓTIMA usando Programação
    Dinâmica clássica (tabela n x W).

    Complexidade de tempo: O(n * W)  -> pseudo-polinomial
    Complexidade de espaço: O(n * W)

    Retorna:
        (valor_otimo, lista_binaria_de_selecao)
    """
    n = instance.n
    W = instance.capacity
    items = instance.items

    # dp[i][w] = melhor valor possível usando os primeiros i itens
    #            com capacidade w
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        it = items[i - 1]
        for w in range(W + 1):
            # opção 1: não inclui o item i
            dp[i][w] = dp[i - 1][w]
            # opção 2: inclui o item i (se couber)
            if it.weight <= w:
                candidate = dp[i - 1][w - it.weight] + it.value
                if candidate > dp[i][w]:
                    dp[i][w] = candidate

    # Reconstrução da solução (quais itens foram escolhidos)
    selection = [0] * n
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selection[i - 1] = 1
            w -= items[i - 1].weight

    return dp[n][W], selection


# ==============================================================================
# 3. METAHEURÍSTICA: ALGORITMO GENÉTICO
# ==============================================================================

class GeneticAlgorithmKnapsack:
    """
    Implementação de um Algoritmo Genético (AG) para o problema da mochila 0/1.

    Representação (codificação):
        Cada indivíduo (cromossomo) é um vetor binário de tamanho n, onde
        cada gene x_i indica se o item i está (1) ou não (0) na mochila.

    Fitness (aptidão):
        Soma dos valores dos itens selecionados, com PENALIZAÇÃO quando a
        soma dos pesos ultrapassa a capacidade (em vez de simplesmente
        descartar o indivíduo, o que reduziria a diversidade da população).

    Operadores genéticos:
        - Seleção: torneio (tournament selection) de tamanho k.
        - Cruzamento (crossover): ponto único (single-point crossover).
        - Mutação: bit-flip com probabilidade p_mut por gene.
        - Elitismo: os melhores indivíduos da geração passam direto
          para a próxima geração, garantindo que a qualidade nunca piora.

    Complexidade: O(gerações * tamanho_populacao * n)
    """

    def __init__(self, instance: KnapsackInstance,
                 population_size: int = 100,
                 generations: int = 200,
                 crossover_rate: float = 0.85,
                 mutation_rate: float = 0.02,
                 tournament_size: int = 3,
                 elitism_size: int = 2,
                 seed: int = 1):
        self.instance = instance
        self.n = instance.n
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.rng = random.Random(seed)

        # histórico de convergência (melhor fitness por geração), útil
        # para gerar o gráfico e discutir os resultados no README.
        self.history_best: List[int] = []
        self.history_avg: List[float] = []

    # ---------------------------------------------------------------- #
    # Funções auxiliares do AG
    # ---------------------------------------------------------------- #

    def _random_individual(self) -> List[int]:
        """Cria um indivíduo aleatório (vetor binário de tamanho n)."""
        return [self.rng.randint(0, 1) for _ in range(self.n)]

    def _fitness(self, individual: List[int]) -> int:
        """
        Calcula o fitness de um indivíduo.

        Se o peso total exceder a capacidade, aplicamos uma penalidade
        proporcional ao excesso de peso, multiplicada por um fator alto,
        de forma que soluções inviáveis sejam sempre piores que soluções
        viáveis, mas ainda participem da evolução (mantendo diversidade
        genética, ao invés de simplesmente eliminá-las).
        """
        total_weight = 0
        total_value = 0
        for gene, item in zip(individual, self.instance.items):
            if gene == 1:
                total_weight += item.weight
                total_value += item.value

        if total_weight <= self.instance.capacity:
            return total_value

        excess = total_weight - self.instance.capacity
        penalty = excess * 10  # fator de penalização
        return max(0, total_value - penalty)

    def _repair(self, individual: List[int]) -> List[int]:
        """
        Estratégia de reparo (usada apenas na solução final apresentada):
        remove itens (começando pelos de pior razão valor/peso) até que
        o indivíduo se torne viável. Isso garante que a solução final
        reportada respeite fisicamente a restrição de capacidade.
        """
        ind = individual[:]
        total_weight = sum(it.weight for g, it in
                            zip(ind, self.instance.items) if g == 1)

        if total_weight <= self.instance.capacity:
            return ind

        # ordena os itens selecionados pela pior razão valor/peso primeiro
        selected_idx = [i for i, g in enumerate(ind) if g == 1]
        selected_idx.sort(
            key=lambda i: self.instance.items[i].value / self.instance.items[i].weight
        )

        for i in selected_idx:
            if total_weight <= self.instance.capacity:
                break
            ind[i] = 0
            total_weight -= self.instance.items[i].weight

        return ind

    def _tournament_selection(self, population: List[List[int]],
                               fitnesses: List[int]) -> List[int]:
        """Seleciona um indivíduo via torneio entre k competidores aleatórios."""
        competitors = self.rng.sample(range(len(population)), self.tournament_size)
        best = max(competitors, key=lambda idx: fitnesses[idx])
        return population[best][:]

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Cruzamento de ponto único (single-point crossover)."""
        if self.rng.random() > self.crossover_rate or self.n < 2:
            return parent1[:], parent2[:]
        point = self.rng.randint(1, self.n - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate(self, individual: List[int]) -> List[int]:
        """Mutação bit-flip: cada gene tem probabilidade mutation_rate de inverter."""
        for i in range(self.n):
            if self.rng.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    # ---------------------------------------------------------------- #
    # Loop principal do Algoritmo Genético
    # ---------------------------------------------------------------- #

    def run(self) -> Tuple[List[int], int]:
        """
        Executa o Algoritmo Genético completo.

        Retorna:
            (melhor_individuo_reparado, seu_valor_total)
        """
        population = [self._random_individual() for _ in range(self.population_size)]

        best_individual = None
        best_fitness = -1

        for gen in range(self.generations):
            fitnesses = [self._fitness(ind) for ind in population]

            # Atualiza o melhor indivíduo global encontrado até agora
            gen_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = population[gen_best_idx][:]

            self.history_best.append(best_fitness)
            self.history_avg.append(sum(fitnesses) / len(fitnesses))

            # --- Elitismo: preserva os melhores indivíduos da geração ---
            ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
            new_population = [population[i][:] for i in ranked[:self.elitism_size]]

            # --- Preenche o restante da nova população via seleção,
            #     cruzamento e mutação ---
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population

        # Repara a melhor solução encontrada, garantindo viabilidade estrita
        best_individual = self._repair(best_individual)
        final_value = sum(it.value for g, it in zip(best_individual, self.instance.items) if g == 1)

        return best_individual, final_value


# ==============================================================================
# 4. FUNÇÕES DE APOIO / RELATÓRIO
# ==============================================================================

def summarize_solution(instance: KnapsackInstance, selection: List[int]) -> dict:
    """Gera um resumo (peso total, valor total, itens escolhidos) de uma solução."""
    chosen_items = [it for g, it in zip(selection, instance.items) if g == 1]
    total_weight = sum(it.weight for it in chosen_items)
    total_value = sum(it.value for it in chosen_items)
    return {
        "total_weight": total_weight,
        "capacity": instance.capacity,
        "total_value": total_value,
        "n_items_chosen": len(chosen_items),
        "chosen_ids": [it.id for it in chosen_items],
    }


def plot_convergence(ga: GeneticAlgorithmKnapsack, optimal_value: int, out_path: str) -> None:
    """Gera e salva o gráfico de convergência do Algoritmo Genético."""
    plt.figure(figsize=(9, 5))
    plt.plot(ga.history_best, label="Melhor fitness (AG)", linewidth=2)
    plt.plot(ga.history_avg, label="Fitness médio da população", linestyle="--", alpha=0.7)
    plt.axhline(y=optimal_value, color="red", linestyle=":",
                label=f"Ótimo exato (DP) = {optimal_value}")
    plt.xlabel("Geração")
    plt.ylabel("Valor (fitness)")
    plt.title("Convergência do Algoritmo Genético - Problema da Mochila 0/1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_results_csv(path: str, exact_value: int, exact_time: float,
                      ga_value: int, ga_time: float) -> None:
    """Salva uma comparação resumida entre o método exato e o AG em CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metodo", "valor_obtido", "tempo_execucao_s", "gap_para_otimo_%"])
        writer.writerow(["Programacao Dinamica (exato)", exact_value, f"{exact_time:.6f}", "0.00"])
        gap = 100 * (exact_value - ga_value) / exact_value if exact_value else 0
        writer.writerow(["Algoritmo Genetico (heuristico)", ga_value, f"{ga_time:.6f}", f"{gap:.2f}"])


# ==============================================================================
# 5. PROGRAMA PRINCIPAL
# ==============================================================================

def main():
    print("=" * 70)
    print("PROBLEMA DA MOCHILA 0/1 - Algoritmo Genetico vs Programacao Dinamica")
    print("=" * 70)

    # 1) Gera (ou poderia carregar de um arquivo) a instância do problema
    instance = generate_instance(n_items=40, capacity_ratio=0.5, seed=42)
    save_instance_json(instance, "results/instance.json")
    print(f"\nInstancia gerada: {instance.n} itens | Capacidade da mochila: {instance.capacity}")

    # 2) Resolve de forma EXATA via Programação Dinâmica (baseline / gabarito)
    t0 = time.time()
    exact_value, exact_selection = solve_exact_dp(instance)
    exact_time = time.time() - t0
    exact_summary = summarize_solution(instance, exact_selection)
    print("\n--- Solucao EXATA (Programacao Dinamica) ---")
    print(f"Valor otimo:        {exact_value}")
    print(f"Peso utilizado:     {exact_summary['total_weight']} / {instance.capacity}")
    print(f"Itens escolhidos:   {exact_summary['n_items_chosen']}")
    print(f"Tempo de execucao:  {exact_time:.6f} s")

    # 3) Resolve de forma HEURÍSTICA via Algoritmo Genético
    ga = GeneticAlgorithmKnapsack(
        instance,
        population_size=100,
        generations=200,
        crossover_rate=0.85,
        mutation_rate=0.02,
        tournament_size=3,
        elitism_size=2,
        seed=1,
    )
    t0 = time.time()
    ga_selection, ga_value = ga.run()
    ga_time = time.time() - t0
    ga_summary = summarize_solution(instance, ga_selection)
    print("\n--- Solucao HEURISTICA (Algoritmo Genetico) ---")
    print(f"Valor encontrado:   {ga_value}")
    print(f"Peso utilizado:     {ga_summary['total_weight']} / {instance.capacity}")
    print(f"Itens escolhidos:   {ga_summary['n_items_chosen']}")
    print(f"Tempo de execucao:  {ga_time:.6f} s")

    gap = 100 * (exact_value - ga_value) / exact_value if exact_value else 0
    print(f"\nGap em relacao ao otimo: {gap:.2f} %")

    # 4) Gera artefatos de saída: gráfico de convergência e CSV comparativo
    plot_convergence(ga, exact_value, "results/convergencia_ag.png")
    save_results_csv("results/comparativo.csv", exact_value, exact_time, ga_value, ga_time)
    print("\nArquivos gerados em 'results/': instance.json, convergencia_ag.png, comparativo.csv")


if __name__ == "__main__":
    main()