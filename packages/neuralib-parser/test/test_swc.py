from types import SimpleNamespace

from neuralib.morpho.swc import SwcNode, SwcFile, SwcPlotOptions


def test_identifier_flags():
    node = SwcNode(1, 1, 0, 0, 0, 1.0, -1)
    assert node.is_soma
    assert not node.is_axon
    assert node.identifier_name == 'soma'
    assert node.point.tolist() == [0, 0, 0]


def test_load_swc_from_mock_file(tmp_path):
    swc_content = """
    1 1 0 0 0 2.0 -1
    2 3 1 0 0 1.0 1
    3 4 0 1 0 1.0 1
    """
    swc_path = tmp_path / "test.swc"
    swc_path.write_text(swc_content, encoding='Big5')

    swc = SwcFile.load(swc_path)
    assert len(swc.node) == 3
    assert swc[1].is_soma
    assert swc['dendrite'].node[0].is_basal_dendrite or swc['dendrite'].node[0].is_apical_dendrite


def test_plot_3d_renders_basal_and_apical_as_dendrites(monkeypatch):
    import sys
    from neuralib.morpho.swc import DEFAULT_COLOR, _plot_swc_3d

    calls = []

    class Plotter:
        def __iadd__(self, obj):
            calls.append(('add', obj))
            return self

        def show(self):
            calls.append(('show', None))

    def spheres(points, r, c):
        calls.append(('spheres', c, len(points)))
        return ('spheres', c)

    def lines(points, c, lw):
        calls.append(('lines', c, len(points)))
        return ('lines', c)

    monkeypatch.setitem(sys.modules, 'vedo', SimpleNamespace(Plotter=Plotter, Spheres=spheres, Lines=lines))
    swc = SwcFile([
        SwcNode(1, 1, 0.0, 0.0, 0.0, 5.0, -1),
        SwcNode(2, 3, 1.0, 0.0, 0.0, 1.0, 1),
        SwcNode(3, 4, 0.0, 1.0, 0.0, 1.0, 1),
    ])

    _plot_swc_3d(swc, radius=True, color=DEFAULT_COLOR)

    assert ('spheres', 'k', 2) in calls
    assert ('lines', 'k', 2) in calls


def test_cli_run_2d_and_3d(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import vedo

    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.setattr(vedo.Plotter, 'show', lambda self: None)

    swc_path = tmp_path / "test_complex.swc"
    swc_content = """
    1 1 0.0 0.0 0.0 5.0 -1
    2 2 10.0 0.0 0.0 1.0 1
    3 2 20.0 0.0 0.0 1.0 2
    4 3 -5.0 -5.0 0.0 1.5 1
    5 3 -10.0 -10.0 0.0 1.2 4
    6 3 -15.0 -10.0 0.0 1.0 5
    7 4 0.0 5.0 5.0 1.5 1
    8 4 0.0 10.0 10.0 1.0 7
    9 4 0.0 15.0 15.0 0.8 8
    """
    swc_path.write_text(swc_content, encoding="Big5")

    # 2D
    monkeypatch.setattr("sys.argv", ["swc", str(swc_path), "--radius", "--2d"])
    SwcPlotOptions().main()

    # 3D
    monkeypatch.setattr("sys.argv", ["swc", str(swc_path), "--radius"])
    SwcPlotOptions().main()

    swc_path.unlink()
