from masthay_helpers.typlotlib import colors_str, pre_colors_dict


def main():
    stars = "*" * 80
    print(f"BEGIN PREDEFINED COLORS\n{stars}\n")
    print(colors_str(colors=pre_colors_dict))
    print(f"END PREDEFINED COLORS\n{stars}\n")

    custom_colors = {
        "red": (1, 0, 0),
        "green": (0, 1, 0),
        "blue": (0, 0, 1),
        "yellow": (1, 1, 0),
        "cyan": (0, 1, 1),
        "magenta": (1, 0, 1),
        "light_gray": (0.75, 0.75, 0.75),
        "dark_gray": (0.25, 0.25, 0.25),
        "white": (1, 1, 1),
        "black": (0, 0, 0),
        "light_blue": (0.5, 0.5, 1),
        "light_green": (0.5, 1, 0.5),
        "light_red": (1, 0.5, 0.5),
        "yellow_green": (0.5, 1, 0.1),
    }
    scaled_custom_colors = {
        k: tuple(int(v * 255) for v in rgb) for k, rgb in custom_colors.items()
    }

    print(f"BEGIN CUSTOM COLORS NORMALIZED\n{stars}\n")
    print(colors_str(colors=custom_colors, normalize=True))
    print(f"END CUSTOM COLORS NORMALIZED\n{stars}\n")

    print(f"BEGIN CUSTOM COLORS SCALED\n{stars}\n")
    print(colors_str(colors=scaled_custom_colors, normalize=False))
    print(f"END CUSTOM COLORS SCALED\n{stars}\n")


if __name__ == "__main__":
    main()
