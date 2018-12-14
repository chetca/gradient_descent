from collections import Counter
from lib.linear_algebra import distance, vector_subtract, scalar_multiply
from functools import reduce
import numpy as np
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def func(v):
    return sum(v_i ** 2 for v_i in v)

def func_gradient(v):
    return [2 * v_i for v_i in v]

def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h

#частное отношение приращений
def partial_difference_quotient(f, v, i, h):
    # вычисляем i-е частное отношение функции f в векторе v
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

#оцениваем градиент
def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)]

def step(v, direction, step_size):
    # двигаемся с шаговым размером step_size в направлении от v
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]

def safe(f):
    #определяем безопасную функцию-обертку для f и возвращаем её
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')    # в Python так обоозначается бесконечность
    return safe_f

# минимизация на основе пакетного градиентного спуска
def bgd(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    # используем градиентный спуск для нахождения вектора theta, 
    # который минимизирует целевую функцию target_fn

    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0 # устанавливаем тета в начальное значение
    target_fn = safe(target_fn) # безопасная версия целевой функции target_fn
    value = target_fn(theta) # минимизируемое значение

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]

        # выбрать то, которое минимизирует функцию ошибок
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        # остановиться, если функция сходится 
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value


"""
# минимизация на основе стохастического градиентного спуска 
def in_random_order(data):
    # генератор, который возвращает элементы данных в случайном порядке
    indexes = [i for i, _ in enumerate(data)] # создаём список индексов
    random.shuffle(indexes) # перемешиваем данные
    for i in indexes: # возвращаем в этом порядке
        yield data[i]

def gen_data(a,b, num = 100, var = 10):
    x = np.array([[1,i] for i in range(num)])
    y = np.dot(x, [a,b]) + np.array([random.normalvariate(0, var) for _ in range(num)])
    return x,y

a = 10
b = 2
x, y = gen_data(a,b)

def sgd(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):

    data = list(zip(x, y))
    theta = theta_0 # первоначальная гипотеза
    alpha = alpha_0 # первоначальный размер шага
    min_theta, min_value = None, float("inf") # минимум на этот момент
    iterations_with_no_improvement = 0

    # останавливаемся, если достигли 100 итераций без улучшений
    while iterations_with_no_improvement < 100:
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data )

        if value < min_value:
            # если найден новый минимум, то запоминаем его
            # и возвращаемся к первоначальному размеру шага
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # в противном случае улучшений нет,
            # поэтому пытаемся сжать размер шага
            iterations_with_no_improvement += 1
            alpha *= 0.9

        # и делаем шаг градиента для каждой из точек данных
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))

    return min_theta
"""

if __name__ == "__main__":

    print("применение градиентного метода")

    np.random.seed(10)
    v = 20*np.random.random((3)) - 10
    print(v)

    tolerance = 0.0000001

    while True:
        #print v, func(v)
        gradient = estimate_gradient(safe(func), v)
        next_v = step(v, gradient, -0.01) # делаем шаг антиградиента
        if distance(next_v, v) < tolerance: # останавливаемся, если сходимся
            break
        #print(v)
        v = next_v # продолжаем, если нет

    print("минимум v", v)
    print("минимальное значение", func(v))
    print()


    print("применение пакетной минимизации bgd")

    v = 20*np.random.random((3)) - 10
    print(v)
    v = bgd(func, func_gradient, v)

    print("минимум v", v)
    print("минимальное значение", func(v))

"""
    print("применение стохастической минимизации sgd")

    v = 20*np.random.random((3)) - 10
    print(v)
    v = sgd(func, func_gradient, x, y, v)

    print("минимум v", v)
    print("минимальное значение", func(v))
"""