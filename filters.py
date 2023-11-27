import numpy as np
import cv2
import matplotlib.pyplot as plt


def median_filter(path_to_image, kernel_size):
    image = cv2.imread(path_to_image)
    return cv2.medianBlur(image, kernel_size)


def sharpening_filter(path_to_image):
    image = cv2.imread(path_to_image)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def gaussian_filter(path_to_image, kernel_size, sigmaX=0):
    """
    Применяет Гауссовский фильтр к изображению.

    :param path_to_image: Путь к изображению.
    :param kernel_size: Размер ядра фильтра (должен быть нечётным).
    :param sigmaX: Стандартное отклонение в направлении X.
                   Если sigmaX равно 0, то оно вычисляется из размера ядра.
    :return: Обработанное изображение.
    """
    image = cv2.imread(path_to_image)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX)


def smoothing_filter(path_to_image, kernel_size):
    """
    Применяет сглаживающий фильтр к изображению.

    :param path_to_image: Путь к изображению.
    :param kernel_size: Размер ядра фильтра.
    :return: Обработанное изображение.
    """
    image = cv2.imread(path_to_image)
    return cv2.blur(image, (kernel_size, kernel_size))


def increase_contrast(path_to_image, gain=1.5, bias=0):
    # Шаг 1: Загрузка изображения
    image = cv2.imread(path_to_image)

    # Шаг 2: Преобразование к формату с плавающей точкой
    float_image = image.astype(float)

    # Шаг 3: Увеличение контраста
    contrasted_image = float_image * gain

    # Шаг 4: Добавление смещения
    contrasted_image += bias

    # Шаг 5: Обрезка значений за пределами [0, 255]
    contrasted_image = np.clip(contrasted_image, 0, 255)

    # Шаг 6: Преобразование обратно в uint8
    return contrasted_image.astype(np.uint8)


def sobel_filter(path_to_image):
    image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return cv2.filter2D(image, -1, kernel)


def prewitt_filter(path_to_image):
    # Чтение изображения
    image = cv2.imread(path_to_image, 0)  # 0 для чтения в градациях серого

    # Определение ядер Прюитта для горизонтальных и вертикальных границ
    prewitt_kernel_horizontal = np.array(
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32
    )

    prewitt_kernel_vertical = np.array(
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32
    )

    # Применение фильтра Прюитта
    edges_horizontal = cv2.filter2D(image, -1, prewitt_kernel_horizontal)
    edges_vertical = cv2.filter2D(image, -1, prewitt_kernel_vertical)

    # Объединение горизонтальных и вертикальных границ
    edges = cv2.addWeighted(edges_horizontal, 0.5, edges_vertical, 0.5, 0)

    return edges


path_to_image = "./image.png"

# filtered_image = sharpening_filter(path_to_image)
# filtered_image = gaussian_filter(path_to_image, 51)
# filtered_image = smoothing_filter(path_to_image, 51)
# filtered_image = increase_contrast(path_to_image)
# filtered_image = sobel_filter(path_to_image)
filtered_image = prewitt_filter(path_to_image)

# показать ихображение
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.show()


# сохранить изображение
# cv2.imwrite("./filtered_image.jpg", filtered_image)
