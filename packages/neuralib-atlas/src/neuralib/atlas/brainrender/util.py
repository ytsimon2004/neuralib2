from brainrender._colors import get_random_colors

__all__ = ['get_color']


def get_color(i: int, color_pattern: str | tuple[str, ...] | list[str]) -> str:
    """
    Get color based on pattern

    :param i: idx of the color pattern list
    :param color_pattern: color pattern list or single element str
    :return:
        color name
    """
    if isinstance(color_pattern, (list, tuple)):
        color_list = list(color_pattern)
        if i >= len(color_list):
            color_list.extend(get_random_colors(i + 2 - len(color_list)))
        return color_list[i]
    elif isinstance(color_pattern, str):
        return color_pattern
    else:
        raise TypeError(f'{type(color_pattern)}')
