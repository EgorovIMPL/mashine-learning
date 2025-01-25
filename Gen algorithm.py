import random

# Параметры задачи
a, b, c, d, e = 1, 2, 3, 4, 5  # Коэффициенты
f = 100  # Правая часть уравнения

# Параметры генетического алгоритма
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
GENERATIONS = 100

#Оценка решения
def check_solution(individual):
    # Вычисляем значение уравнения
    value = a * individual[0] + b * individual[1] + c * individual[2] + d * individual[3] + e * individual[4]
    # Мы хотим минимизировать абсолютную разницу с f
    return abs(value - f)

#Создание начальной популяции
def create_population(size):
    return [[random.randint(1, 86), random.randint(1, 43), random.randint(1, 29), random.randint(1, 22), random.randint(1, 18)] for _ in range(size)]

#Выбор родителей на основе их пригодности
def select_parents(population):
    population.sort(key=check_solution)  # Сортируем по пригодности
    return population[:2]  # Возвращаем двух лучших

#Создание детей
def create_child(parent1, parent2):
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

#Мутация детей
def mutate(child):
    for i in range(len(child)):
        if random.random() < MUTATION_RATE:
            child[i] += random.randint(1, 3)  # Изменяем значение на 1, 2 или 3
    return child

#Основной цикл генетического алгоритма
def genetic_algorithm():
    population = create_population(POPULATION_SIZE)
    
    for generation in range(GENERATIONS):
        new_population = []
        
        for _ in range(POPULATION_SIZE):
            parent1, parent2 = select_parents(population)
            child = create_child(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
        
    # Находим лучшее решение после всех поколений
    best_solution = min(population, key=check_solution)
    return best_solution

# Запуск генетического алгоритма
best_solution = genetic_algorithm()
print("Лучшее решение найдено:", best_solution)
print("Разница лучшего решения с ответом:", check_solution(best_solution))
