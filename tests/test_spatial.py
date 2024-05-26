"""Test spatial methods."""

from gridit import Grid
from gridit.spatial import flat_grid_intersect, is_same_crs

from .conftest import requires_pkg


def test_is_same_crs():
    assert is_same_crs("EPSG:2193", "EPSG:2193")
    assert not is_same_crs("EPSG:2193", "EPSG:2194")
    assert is_same_crs("this is same", "this is similar")
    assert not is_same_crs("this is one", "quite different, really")


@requires_pkg("shapely")
def test_flat_grid_intersect():
    # exact overlap, same resolution
    A = Grid(10.0, (6, 8), (1000, 2000.0))
    AAv_l = list(flat_grid_intersect(A, A, "vector"))
    exp_l = list(((idx, 1.0), (idx, 1.0)) for idx in range(48))
    assert AAv_l == exp_l
    AAr_s = set(flat_grid_intersect(A, A, "raster"))
    assert set(exp_l) == AAr_s

    # exact overlap, double resolution
    B = Grid(20.0, (3, 4), (1000, 2000.0))
    assert A.bounds == B.bounds
    assert A.resolution * 2 == B.resolution
    ABv_s = set(flat_grid_intersect(A, B, "vector"))
    assert ((0, 1.0), (0, 0.25)) in ABv_s
    exp_A_in_B = set(range(48))
    exp_B_in_A = set(range(12))
    assert {itm[0][0] for itm in ABv_s} == exp_A_in_B
    assert {itm[1][0] for itm in ABv_s} == exp_B_in_A
    assert {itm[0][1] for itm in ABv_s} == {1.0}
    assert {itm[1][1] for itm in ABv_s} == {0.25}
    assert len(ABv_s) == 48
    ABr_s = set(flat_grid_intersect(A, B, "raster"))
    assert ABv_s == ABr_s
    BAv_s = set(flat_grid_intersect(B, A, "vector"))
    assert ((0, 0.25), (0, 1.0)) in BAv_s
    assert {itm[0][0] for itm in BAv_s} == exp_B_in_A
    assert {itm[1][0] for itm in BAv_s} == exp_A_in_B
    assert {itm[0][1] for itm in BAv_s} == {0.25}
    assert {itm[1][1] for itm in BAv_s} == {1.0}
    assert len(BAv_s) == 48
    BAr_s = set(flat_grid_intersect(B, A, "raster"))
    assert BAv_s == BAr_s

    # shift grid, no overlap, same resolution
    B = Grid(10.0, (3, 4), (2000, 3000.0))
    assert A.bounds != B.bounds
    assert A.resolution == B.resolution
    assert set(flat_grid_intersect(A, B, "vector")) == set()
    assert set(flat_grid_intersect(A, B, "raster")) == set()
    assert set(flat_grid_intersect(B, A, "vector")) == set()
    assert set(flat_grid_intersect(B, A, "raster")) == set()

    # shift grid, some overlap, same resolution
    B = Grid(10.0, (6, 8), (1030, 1980.0))
    assert A.bounds != B.bounds
    assert A.resolution == B.resolution
    ABv_s = set(flat_grid_intersect(A, B, "vector"))
    assert ((19, 1.0), (0, 1.0)) in ABv_s
    exp_A_in_B = {
        19,
        20,
        21,
        22,
        23,
        27,
        28,
        29,
        30,
        31,
        35,
        36,
        37,
        38,
        39,
        43,
        44,
        45,
        46,
        47,
    }
    exp_B_in_A = {
        0,
        1,
        2,
        3,
        4,
        8,
        9,
        10,
        11,
        12,
        16,
        17,
        18,
        19,
        20,
        24,
        25,
        26,
        27,
        28,
    }
    assert {itm[0][0] for itm in ABv_s} == exp_A_in_B
    assert {itm[1][0] for itm in ABv_s} == exp_B_in_A
    assert {itm[0][1] for itm in ABv_s} == {1.0}
    assert {itm[1][1] for itm in ABv_s} == {1.0}
    assert len(ABv_s) == 20
    ABr_s = set(flat_grid_intersect(A, B, "raster"))
    assert ABv_s == ABr_s
    BAv_s = set(flat_grid_intersect(B, A, "vector"))
    assert ((0, 1.0), (19, 1.0)) in BAv_s
    assert {itm[0][0] for itm in BAv_s} == exp_B_in_A
    assert {itm[1][0] for itm in BAv_s} == exp_A_in_B
    assert {itm[0][1] for itm in BAv_s} == {1.0}
    assert {itm[1][1] for itm in BAv_s} == {1.0}
    assert len(BAv_s) == 20
    BAr_s = set(flat_grid_intersect(B, A, "raster"))
    assert BAv_s == BAr_s

    # exact overlap, resolution is not a multiple
    B = Grid(6.0 + 2 / 3.0, (9, 12), (1000, 2000.0))
    assert A.bounds == B.bounds
    assert A.resolution * 2.0 / 3.0 == B.resolution
    ABv_s = set(flat_grid_intersect(A, B, "vector"))
    exp_A_in_B = set(range(48))
    exp_B_in_A = set(range(108))
    assert {itm[0][0] for itm in ABv_s} == exp_A_in_B
    assert {itm[1][0] for itm in ABv_s} == exp_B_in_A
    assert {round(itm[0][1], 2) for itm in ABv_s} == {0.11, 0.22, 0.44}
    assert {round(itm[1][1], 2) for itm in ABv_s} == {0.25, 0.5, 1.0}
    assert len(ABv_s) == 192
    ABr_s = set(flat_grid_intersect(A, B, "raster"))
    assert ABv_s != ABr_s
    assert {itm[0][0] for itm in ABr_s} == exp_A_in_B
    assert {itm[1][0] for itm in ABr_s} == exp_B_in_A
    assert {round(itm[0][1], 2) for itm in ABr_s} == {0.44}
    assert {itm[1][1] for itm in ABr_s} == {1.0}
    BAv_s = set(flat_grid_intersect(B, A, "vector"))
    assert {itm[0][0] for itm in BAv_s} == exp_B_in_A
    assert {itm[1][0] for itm in BAv_s} == exp_A_in_B
    assert {round(itm[0][1], 2) for itm in BAv_s} == {0.25, 0.5, 1.0}
    assert {round(itm[1][1], 2) for itm in BAv_s} == {0.11, 0.22, 0.44}
    assert len(BAv_s) == 192
    BAr_s = set(flat_grid_intersect(B, A, "raster"))
    assert BAv_s != BAr_s
    assert {itm[0][0] for itm in BAr_s} == exp_B_in_A
    assert {itm[1][0] for itm in BAr_s} == exp_A_in_B
    assert {itm[0][1] for itm in BAr_s} == {1.0}
    assert {round(itm[1][1], 2) for itm in BAr_s} == {0.44}

    # mostly overlap, resolution slightly different
    B = Grid(9.0, (6, 8), (1001, 1999))
    assert A.bounds != B.bounds
    assert A.resolution != B.resolution
    ABv_s = set(flat_grid_intersect(A, B, "vector"))
    exp_A_in_B = set(range(48))
    assert {itm[0][0] for itm in ABv_s} == exp_A_in_B
    assert {itm[1][0] for itm in ABv_s} == exp_A_in_B
    ABv_f1 = {round(itm[0][1], 2) for itm in ABv_s}
    assert (len(ABv_f1), min(ABv_f1), max(ABv_f1)) == (36, 0.01, 0.81)
    ABv_f2 = {round(itm[1][1], 2) for itm in ABv_s}
    assert (len(ABv_f2), min(ABv_f2), max(ABv_f2)) == (36, 0.01, 1.0)
    assert len(ABv_s) == 140
    ABr_s = set(flat_grid_intersect(A, B, "raster"))
    assert ABv_s != ABr_s
    assert exp_A_in_B.difference({itm[0][0] for itm in ABr_s}) == {
        7,
        15,
        23,
        31,
        39,
        47,
    }
    assert {itm[1][0] for itm in ABr_s} == exp_A_in_B
    assert {round(itm[0][1], 2) for itm in ABr_s} == {0.81}
    assert {itm[1][1] for itm in ABr_s} == {1.0}
    BAv_s = set(flat_grid_intersect(B, A, "vector"))
    assert {itm[0][0] for itm in BAv_s} == exp_A_in_B
    assert {itm[1][0] for itm in BAv_s} == exp_A_in_B
    BAv_f1 = {round(itm[0][1], 2) for itm in BAv_s}
    assert (len(BAv_f1), min(BAv_f1), max(BAv_f1)) == (36, 0.01, 1.0)
    BAv_f2 = {round(itm[1][1], 2) for itm in BAv_s}
    assert (len(BAv_f2), min(BAv_f2), max(BAv_f2)) == (36, 0.01, 0.81)
    assert len(BAv_s) == 140
    BAr_s = set(flat_grid_intersect(B, A, "raster"))
    assert BAv_s != BAr_s
    assert {itm[0][0] for itm in BAr_s} == exp_A_in_B
    assert exp_A_in_B.difference({itm[1][0] for itm in BAr_s}) == {
        7,
        15,
        23,
        31,
        39,
        47,
    }
    assert {itm[0][1] for itm in BAr_s} == {1.0}
    assert {round(itm[1][1], 2) for itm in BAr_s} == {0.81}
