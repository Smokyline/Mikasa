import tensorflow as tf

# объявляем константу a. Это константа и её значение будет зашито в самом графе
# в объявлении ниже указаны все возможные параметры, хотя достаточно было указать только значение:
# a = tf.constant(2.0)
# описание параметров:
# value (первый аргумент) - значение константы
# shape - размерность. Например: [] - число, [5] - массив из 5 элементов, [2, 3] - матрица 2x3(2 строки на 3 столбца)
# dtype - используемый тип данных, список возможных значений тут https://www.tensorflow.org/api_docs/python/tf/DType
# name - имя узла. Позволяет дать узлу имя и в дальнейшем находить узел по нему
a = tf.constant(2.0, shape=[], dtype=tf.float32, name="a")
# объявляем переменную x
# при объявлении переменной можно указать достаточно много аргументов
# на полный список можно взглянуть в документации, скажу только про основные:
# initial_value - значение переменной после инициализации
# dtype - тип, name - имя, как и у констант
x = tf.Variable(initial_value=3.0, dtype=tf.float32)
# поскольку обычно нам нужно передавать в модель данные по ходу работы, константы нам не очень подходят
# для входных данных предусмотрен специальный тип placeholder
# в отличии от константы он не требует указать значение заранее, но требует указать тип
# также можно указать размерность и имя
b = tf.placeholder(tf.float32, shape=[])
# и объявляем саму операцию умножения, при желании можно так же указать имя
f = tf.add(tf.multiply(a, x), b)  # можно было написать просто f = a*x + b
#f = a*x+b  # можно было написать просто f = a*x + b

#tf.device('/gpu:0')

with tf.Session() as session:
    # прежде всего нужно инициализировать все глобальные переменные
    # в нашем случае это только x
    tf.global_variables_initializer().run()
    # просим вычислить значение узла f внутри сессии
    # в параметре feed_dict передаём значения всех placeholder'ов
    # в данном случае b = -5
    # функция вернёт список значений всех узлов, переданных на выполнение
    result_f, result_a, result_x, result_b = session.run([f, a, x, b], feed_dict={b: -5})
    print("f = %.1f * %.1f + %.1f = %.1f" % (result_a, result_x, result_b, result_f))
    print("a = %.1f" % a.eval())  # пока сессия открыта, можно вычислять узлы
    # метод eval похож на метод run у сессии, но не позволяет передать входные данные (параметр feed_dict)

    # переменные можно модифицировать во время выполнения, не трогая граф:
    x = x.assign_add(1.0)
    print("x = %.1f" % x.eval())

    # Вывод:
    # f = 2.0 * 3.0 + -5.0 = 1.0
    # a = 2.0
    # x = 4.0