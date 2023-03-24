import random
import os


def generate_data(filename, width, height, n_points, intensity_range):
    with open(filename, "w+") as f:
        f.write(f"{str(width).ljust(10)} {height}\n")
        for i in range(n_points):
            x = int(width * random.random())
            y = int(height * random.random())
            intensity = random.random() * (intensity_range[1] - intensity_range[0]) + intensity_range[0]
            f.write(f"{str(x).ljust(10)} {str(y).ljust(10)} {str(round(intensity, 3)).ljust(10)}\n")


def main():
    os.makedirs("data", exist_ok=True)
    for i in range(9):
        generate_data(f"data/heat_points_{i}", 2 ** (i + 3), 2 ** (i + 3), 50, (2, 5))


if __name__ == "__main__":
    main()
