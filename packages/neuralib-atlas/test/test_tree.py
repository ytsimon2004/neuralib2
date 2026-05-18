import plotly.graph_objects as go

from neuralib.atlas.plot import plot_sunburst_acronym, print_tree

_SUPPRESS_SHOW = True


def test_plot_sunburst_acronym_uses_paired_columns(monkeypatch):
    if _SUPPRESS_SHOW:
        monkeypatch.setattr(go.Figure, 'show', lambda self: None)

    plot_sunburst_acronym()


def test_print_tree():

    def print_all():
        print_tree()

    def print_node():
        print_tree('VIS')

    print_all()
    print_node()
