# Импорт библиотек
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


def load_distance_matrix(filename=None, num_points=5):
    """
    Функция загрузки данных.
    Если указан filename, загружает матрицу расстояний из CSV-файла.
    Иначе генерирует случайную симметричную матрицу расстояний размером num_points x num_points.

    Возвращает: матрицу расстояний в виде списка списков.
    """
    if filename:
        # Загрузка из CSV файла
        matrix = np.loadtxt(filename, delimiter=',')
        return matrix.tolist()
    else:
        # Генерация случайной симметричной матрицы с нулями на диагонали
        rng = np.random.default_rng()
        mat = rng.integers(10, 100, size=(num_points, num_points))
        mat = (mat + mat.T) // 2  # сделать симметричной
        np.fill_diagonal(mat, 0)
        return mat.tolist()


def build_routing_model(distance_matrix):
    """
    Функция построения модели задачи коммивояжера.
    """
    data = {}
    data['distance_matrix'] = distance_matrix
    data['num_vehicles'] = 1
    data['depot'] = 0  # стартовая точка маршрута

    manager = pywrapcp.RoutingIndexManager(len(distance_matrix),
                                           data['num_vehicles'], data['depot'])

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """
        Функция стоимости перехода между узлами.
        Получает индексы в модели и возвращает расстояние между соответствующими пунктами.
        """
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Устанавливаем функцию стоимости для всех транспортных средств (одного)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Параметры поиска решения: используем эвристику "самый дешёвый дуговой путь"
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    return manager, routing, search_parameters


def solve_tsp(manager, routing, search_parameters):
    """
    Функция решения задачи коммивояжера.

    Запускает оптимизатор и извлекает оптимальный маршрут из решения.

    Возвращает список индексов пунктов в порядке посещения или None при отсутствии решения.
    """
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        route = []
        index = routing.Start(0)

        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))

        route.append(manager.IndexToNode(index))  # возвращаемся в депо

        return route
    else:
        return None


def visualize_route(distance_matrix, route):
    """
    Функция визуализации оптимального маршрута на графе.

    Использует networkx для построения графа и matplotlib для отображения.

    Аргументы:
      - distance_matrix: матрица расстояний (список списков)
      - route: список индексов пунктов в порядке посещения
    """

    G = nx.Graph()

    n = len(distance_matrix)

    # Добавляем ребра с весами (расстояниями) между всеми узлами
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=distance_matrix[i][j])

    pos = nx.spring_layout(G)  # расположение узлов для визуализации

    # Рисуем узлы и подписи к ним
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos)

    # Рисуем все рёбра серым цветом с прозрачностью
    nx.draw_networkx_edges(G, pos, alpha=0.3)

    # Формируем список рёбер маршрута для выделения красным цветом
    path_edges = list(zip(route[:-1], route[1:]))

    nx.draw_networkx_edges(G, pos,
                           edgelist=path_edges,
                           edge_color='r',
                           width=3)

    plt.title("Оптимальный маршрут коммивояжера")
    plt.axis('off')
    plt.show()


def main():
    """
    Основная функция запуска программы:
     - загружает данные,
     - строит модель,
     - решает задачу,
     - выводит результат и визуализирует маршрут.
     """

    # Пример: загрузка из файла 'distances.csv' или генерация случайных данных с 6 пунктами
    filename = None  # 'distances.csv' или None
    num_points = 6

    distance_matrix = load_distance_matrix(filename=filename, num_points=num_points)

    print("Матрица расстояний:")
    for row in distance_matrix:
        print(row)

    manager, routing, search_parameters = build_routing_model(distance_matrix)

    route = solve_tsp(manager, routing, search_parameters)

    if route:
        print("\nОптимальный маршрут:", route)
        visualize_route(distance_matrix, route)
    else:
        print("Решение не найдено.")


if __name__ == '__main__':
    main()
