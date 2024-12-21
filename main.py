from math import sqrt

from sympy import Matrix, sin, eye, integrate, solve, expand, collect
from sympy.abc import x, y, l, a

from matplotlib import pyplot


def _fill_matrix(omega_y: list, u_y: list) -> Matrix:
    """
    Заполнение матрицы для решения однородной системы
    и нахождения 
    """

    # Размерность матрицы
    matrix_dim = len(omega_y)
    # Матрица с единицами на главной диагонали
    c = eye(matrix_dim)
    for i in range(matrix_dim):
        for j in range(matrix_dim):
                c_i_j = (-l * integrate(expand(omega_y[j] * u_y[i]), (y, 0, 1))).evalf()
                c[i, j] += c_i_j
    
    return c


def _calc_det(c: Matrix):
    """
    Подсчет определителя матрицы
    """

    # Количество строк и столбцов
    dim_row, dim_col = c.shape

    # Определитель можно считать только у квадратных матриц =)
    if dim_row != dim_col:
        raise Exception("Only square matrices are allowed!")

    # Размерность 
    dim = dim_row

    if dim == 1:
        raise Exception("Matrix dimensity is too small")

    # Определитель матрицы размерности 2
    if dim == 2:
        return c.det()
    
    # Будем искать определитель через разложение по столбцу
    det = None
    for i in range(dim):
        # Вычеркиваем строку и столбец
        c_next = c.copy()
        c_next.row_del(i)
        c_next.col_del(0)
        
        # Считаем следующий определитель
        inner_matrix_det = expand(c[i, 0] * _calc_det(c_next))

        # Учитываем знак, который при разложении вылезает
        if det is None:
            det = inner_matrix_det
        else:
            det += inner_matrix_det if i % 2 == 0 else (-1) * inner_matrix_det

    return det


def _subs_value_in_matrix(c: Matrix, sub_symbol, value: float):
    """
    Подстановка значения value вместо символа sub_symbol во все
    элементы матрицы
    """
    dim_row, dim_col = c.shape

    if dim_row != dim_col:
        raise Exception("Only square matrices are allowed!")
    dim = dim_row

    subbed_matrix = []
    for i in range(dim):
        row = c.row(i)
        subbed_row = [item.subs(sub_symbol, value) for item in row]
        subbed_matrix.append(subbed_row)
    
    return Matrix(subbed_matrix)


def _solve_linear_system(c: Matrix, b: Matrix) -> list[float]:
    """
    Решение системы линейных уравнений, задающейся матрицей 'c' и
    столбцом свободных членов 'b'
    """

    # Решаю систему, решение в последнем столбце
    c_augmented = c.row_join(b)
    # Решаю систему, решение в последнем столбце
    c_eye, _ = c_augmented.rref()
    dim = c_eye.shape[0]
    # Решение системы при заданном 'b'
    roots = [c_eye[row, dim] for row in range(c_eye.rows)]

    return roots


def _calc_eigen_function(omega_i: list, c_i: list[float]):
    """
    Подсчёт собственной функции
    """

    # Будущая собственная функция
    u_i = 0
    dim = len(omega_i)
    for i in range(dim):
        u_i += omega_i[i] * c_i[i]

    # Нормируем полученную функцию
    u_i /= sqrt(
        integrate(expand(u_i * u_i), (y, 0, 1)).evalf()
    ).real

    return u_i


def _calc_kernel(omega_y: list, u_y: list):
    """
    Подсчет конечного ядра
    """

    K_x_y = 0
    dim = len(omega_y)
    for i in range(dim):
        u_x_i = u_y[i].subs(y, x)
        K_x_y += omega_y[i] * u_x_i
    
    return(K_x_y)


def _tab_func(func, symbol, arg_start: float, arg_end: float, n: int) -> tuple[list, list]:
    """
    Табуляция функции 'func' в пределах отрезка [arg_start, arg_end]
    :param func: Выражение sympy по неизвестной 'symbol'
    :param symbol: Символ sympy, являющийся неизвестной
    :param arg_start: Начало отрезка табуляции
    :param arg_end: Конец отрезка табуляции
    :param n: На сколько частей разбить отрезок табуляции
    :result: Два списка:
        - список с точками табуляции
        - список со значениями функции в точках
    """

    arg_step: float = (arg_end - arg_start) / n
    arg: float = arg_start
    points: list[float] = []
    func_values: list[float] = []
    while arg_start <= arg <= arg_end:
        points.append(arg)
        func_values.append(func.subs(symbol, arg).evalf())
        arg += arg_step

    return points, func_values


def main():
    # Из условия варианта
    omega_y: list = [sin(y), sin(2*y), sin(3*y), y**2, y**4]
    u_y: list = [sin(y), 2 * sin(2*y), sin(3*y), y**2, y**4]

    # Списки в которых будут собственные значения и собственные функции после их нахождения
    eigenvalues = []
    eigen_functions = []

    H_x_y = _calc_kernel(omega_y, u_y)

    # Т.к. имеем дело с вырожденным ядром - мы знаем, сколько у него 
    # будет пар (собственное значение - собственная функция)
    for k in range(len(omega_y)):
        # Сначала решаем однородную систему

        # Заполняется матрица, где лямбда - неизвестная
        c = _fill_matrix(omega_y, u_y)

        # Чтобы найти лямбды надо записать определитель
        # и решить характеристическое уравнение
        characteristic_poly = _calc_det(c)
        lambdas = solve(characteristic_poly)
        # print(f'Характеристическое уравнение: {characteristic_poly}')

        # Теперь отсортируем, найдем самое маленькое по модулю
        lambdas.sort(key=lambda item: abs(item))
        lambda_i = lambdas[0]
        eigenvalues.append(lambda_i)
        # print(f'Собственные значения: {lambdas}')

        # Подставляю найденное наименьшее значение вместо лямбды
        c = _subs_value_in_matrix(c, l, lambda_i)

        # Решаю систему уравнений с заданным 'b'
        roots = _solve_linear_system(c, Matrix([0, 0, 0, 0, 1]))
        # print(f'Решения системы: {roots}')

        # Считаю собственную функцию, соответствующую данному собственному значению
        u_y_i = _calc_eigen_function(omega_y, roots)
        eigen_functions.append(u_y_i)
        # print(u_y_i)

        # Шаг уменьшение ядра
        u_x_i = u_y_i.subs(y, x)
        H_x_y -= u_x_i * u_y_i / lambda_i
        H_x_y = expand(H_x_y)
        # Собраем коэффициенты при функциях для новой итерации
        for func in omega_y:
            H_x_y = collect(H_x_y, func)
        for i, omega in enumerate(omega_y):
            u_y[i] = H_x_y.coeff(omega).subs(x, y)
    
    

    # Вывод собственных значений и собственных функций
    print("Собственные значения: ")
    for i, l_val in enumerate(eigenvalues, start=1):
        print(f'\t{i}. {l_val}')
    print("Собственные функции: ")
    for i, func in enumerate(eigen_functions, start=1):
        print(f'\t{i}. {func}')

    # Неоднородность по условию
    f = sin(3*x) + a * x ** 4
    print("Условия разрешимости при полученных собственных значениях: ")
    for i, func in enumerate(eigen_functions, start=1):
        integral = integrate(
            expand(f * func.subs(y, x)), 
            (x, 0, 1)
        ).evalf()
        print(f'\t{i}. a = {solve(integral)}')

    # Построение графиков собственных функций
    arg_start = 0
    arg_end = 1
    n = 1e4
    for i, func in enumerate(eigen_functions, start=1):
        points, func_values = _tab_func(func, y, arg_start, arg_end, n)
        pyplot.plot(points, func_values, label=f'u_{i}')
    
    pyplot.legend(loc='upper left')
    pyplot.show()


if __name__ == "__main__":
    main()