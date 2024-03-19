from PIL import Image, ImageDraw
import math
import numpy as np


def is_black(pixel):
    return pixel == 0


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def dfs(x, y, visited, adjacent_pixels):
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    visited.add((x, y))
    adjacent_pixels.append((x, y))

    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if (0 <= new_x < width and
                0 <= new_y < height and
                (new_x, new_y) not in visited and
                is_black(pixels[new_x, new_y])):
            dfs(new_x, new_y, visited, adjacent_pixels)


def extract_adjacent_pixels(start_x, start_y):
    visited = set()
    adjacent_pixels = []
    dfs(start_x, start_y, visited, adjacent_pixels)
    return adjacent_pixels


def fit_line_ransac(points, iterations=20, threshold=5, point_pect=0.75):
    best_line = None
    best_inliers = []
    num_points = len(points)

    for _ in range(iterations):
        sample_indices = np.random.choice(num_points, 2, replace=False)
        sample_points = [points[i] for i in sample_indices]
        x1, y1 = sample_points[0]
        x2, y2 = sample_points[1]

        if (x1==x2):
            m, c = 0, 0
            distances = [abs(x - x1) for x, y in points]
        else:
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
            distances = [abs(m * x - y + c) / np.sqrt(m ** 2 + 1) for x, y in points]

        inliers = [points[i] for i, distance in enumerate(distances) if distance < threshold]

        if len(inliers) > len(best_inliers):
            best_line = (m, c)
            best_inliers = inliers

        if len(best_inliers) < len(points)*point_pect:
            return False
        else:
            return True


def main():
    image = Image.open("test.png")
    image_draw = image.copy()
    gray_image = image.convert('L')
    global width, height
    width, height = gray_image.size
    line_positions = []
    global pixels
    pixels = gray_image.load()
    line_tmp = []
    line_result = []

    for y in range(height):
        for x in range(width):
            if is_black(pixels[x, y]):
                line_tmp.append((x, y))

    while len(line_tmp) > 10:
        start_x, start_y = line_tmp[0][0], line_tmp[0][1]
        adjacent_pixels = extract_adjacent_pixels(start_x, start_y)
        line_tmp = [coord for coord in line_tmp if coord not in adjacent_pixels]
        line_positions.append(adjacent_pixels)

    for i in range(len(line_positions)):
        if fit_line_ransac(line_positions[i]):
            line_positions[i].sort()
            line_result.append(line_positions[i])

    draw = ImageDraw.Draw(image_draw)
    color = (0, 255, 0)
    thickness = 2

    for i in range(len(line_result)):
        draw.line((line_result[i][0], line_result[i][-1]), fill=color, width=thickness)

    print("直线共计{}条".format(len(line_result)))
    print("直线位置和长度：")
    for line in line_result:
        start_point = line[0]
        end_point = line[-1]
        length = np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
        print("起点：{}，终点：{}，长度：{:.2f}".format(start_point, end_point, length))

    image_draw.save('result.png')
    image_draw.show()


if __name__ == "__main__":
    main()



