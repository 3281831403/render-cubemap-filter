"""render-cubemap-filter 的 15 条红测。

每条测试都至少调用一个 ``cubemap.prefilter`` 中的公开桩。初始桩统一抛
``NotImplementedError``，因此应恰好得到 15 failed，不能出现收集错误。
"""

import math

import pytest

from cubemap.prefilter import (
    CubemapError,
    dir_to_face_uv,
    ggx_sample,
    mip_for_roughness,
    prefilter_batch,
    prefilter_face,
    seam_fix,
    texel_angle,
    validate_cubemap,
)

TOL = 1e-6


def _face(size=8, value=0.5):
    """构造均匀值的 ``size × size`` 面。"""
    return [[float(value) for _ in range(size)] for _ in range(size)]


def _faces(size=8, value=0.2):
    """构造 6 个均匀值的面。"""
    return [_face(size, value) for _ in range(6)]


def _cubemap(size=8, value=0.2):
    return {"faces": _faces(size, value)}


def _gradient_face(size=8):
    """构造逐纹素变化的面，用于检验恒等性。"""
    total = float(size * size)
    return [
        [((row * size + col) + 0.5) / total for col in range(size)]
        for row in range(size)
    ]


def _clone_faces(faces):
    return [[row[:] for row in face] for face in faces]


def _clone_result(result):
    return {
        "cubemaps": [
            {"faces": _clone_faces(item["faces"])} for item in result["cubemaps"]
        ],
        "count": result["count"],
    }


def _assert_face_close(got, want, tol=TOL):
    assert len(got) == len(want)
    for got_row, want_row in zip(got, want):
        assert len(got_row) == len(want_row)
        for got_value, want_value in zip(got_row, want_row):
            assert got_value == pytest.approx(want_value, abs=tol)


def _assert_result_close(got, want):
    assert got["count"] == want["count"]
    assert len(got["cubemaps"]) == len(want["cubemaps"])
    for got_item, want_item in zip(got["cubemaps"], want["cubemaps"]):
        assert len(got_item["faces"]) == len(want_item["faces"])
        for got_face, want_face in zip(got_item["faces"], want_item["faces"]):
            _assert_face_close(got_face, want_face)


# 缺陷 1：方向不归一，面归属错。
def test_direction_is_normalized_before_face_selection():
    axes = [
        ((1.0, 0.0, 0.0), 0),
        ((-1.0, 0.0, 0.0), 1),
        ((0.0, 1.0, 0.0), 2),
        ((0.0, -1.0, 0.0), 3),
        ((0.0, 0.0, 1.0), 4),
        ((0.0, 0.0, -1.0), 5),
    ]
    for direction, expected_face in axes:
        face, u, v = dir_to_face_uv(direction)
        assert face == expected_face
        assert 0.0 <= u <= 1.0
        assert 0.0 <= v <= 1.0

    reference = dir_to_face_uv((0.3, -0.4, 1.0))
    for scale in (1e-9, 0.01, 0.25, 3.0, 17.0, 1e9):
        scaled = dir_to_face_uv((0.3 * scale, -0.4 * scale, 1.0 * scale))
        assert scaled[0] == reference[0]
        assert scaled[1] == pytest.approx(reference[1], abs=TOL)
        assert scaled[2] == pytest.approx(reference[2], abs=TOL)

    for scale in (1e-9, 1e9):
        face, u, v = dir_to_face_uv((0.0, 0.0, scale))
        assert face == 4
        assert u == pytest.approx(0.5, abs=TOL)
        assert v == pytest.approx(0.5, abs=TOL)

    with pytest.raises(CubemapError):
        dir_to_face_uv((0.0, 0.0, 0.0))


# 缺陷 2：纹素立体角用平面像素面积，mip 偏小。
def test_texel_angle_uses_solid_angle_not_planar_area():
    assert texel_angle(1) == pytest.approx(2.0 * math.pi / 3.0, abs=TOL)
    for size in (2, 8, 64):
        assert 6 * size * size * texel_angle(size) == pytest.approx(
            4.0 * math.pi, abs=1e-9
        )
        # 平面像素面积 (2 / size)² 会高估纹素立体角。
        assert texel_angle(size) < (2.0 / size) ** 2
    assert texel_angle(2) == pytest.approx(texel_angle(1) / 4.0, rel=1e-9)

    with pytest.raises(CubemapError):
        texel_angle(0)


# 缺陷 3：mip 选择向下取整，粗糙面锐利。
def test_mip_selection_rounds_up():
    assert mip_for_roughness(0.0, 6) == 0
    assert mip_for_roughness(1.0, 6) == 6
    assert mip_for_roughness(0.3, 6) == 2  # 连续级别 1.8，向上取整
    assert mip_for_roughness(0.5, 6) == 3
    assert mip_for_roughness(0.01, 6) == 1

    levels = [mip_for_roughness(step / 20.0, 6) for step in range(21)]
    assert levels == sorted(levels)
    assert all(0 <= level <= 6 for level in levels)
    assert mip_for_roughness(0.3, 6) > 0.3 * 6.0

    with pytest.raises(CubemapError):
        mip_for_roughness(-0.1, 6)
    with pytest.raises(CubemapError):
        mip_for_roughness(0.5, -1)


# 缺陷 4：GGX 采样不按粗糙度，全镜面。
def test_ggx_sampling_tracks_roughness():
    normal = (0.0, 0.0, 1.0)
    mirror = [ggx_sample(normal, 0.0, (step / 16.0, 0.5)) for step in range(16)]
    for half_vector in mirror:
        for got, want in zip(half_vector, normal):
            assert got == pytest.approx(want, abs=TOL)

    xis = [(step / 32.0, ((step * 7) % 32) / 32.0) for step in range(32)]
    smooth = [ggx_sample(normal, 0.05, xi) for xi in xis]
    rough = [ggx_sample(normal, 0.9, xi) for xi in xis]
    smooth_dot = sum(half[2] for half in smooth) / len(smooth)
    rough_dot = sum(half[2] for half in rough) / len(rough)

    assert smooth_dot > 0.999
    assert rough_dot < 0.9
    assert rough_dot < smooth_dot
    for half_vector in smooth + rough:
        length = math.sqrt(sum(component * component for component in half_vector))
        assert length == pytest.approx(1.0, abs=TOL)
        assert half_vector[2] >= -TOL

    assert ggx_sample(normal, 0.4, (0.25, 0.75)) == ggx_sample(
        normal, 0.4, (0.25, 0.75)
    )
    assert ggx_sample(normal, 0.4, (0.25, 0.75)) != ggx_sample(
        normal, 0.4, (0.75, 0.25)
    )


# 缺陷 5：滤波不按样本数归一，亮度随样本数变化。
def test_prefilter_averages_over_sample_count():
    face = _face(8, 0.5)
    for samples in (1, 4, 16):
        result = prefilter_face(face, 0.7, samples)
        assert len(result) == 8
        assert all(len(row) == 8 for row in result)
        for row in result:
            for value in row:
                assert value == pytest.approx(0.5, abs=TOL)


# 缺陷 6：面边缘重复纹素，接缝亮线。
def test_seam_fix_removes_bright_edge_line():
    faces = _faces(8, 0.2)
    for row in range(8):
        faces[0][row][0] = 0.6  # +X 面左边缘的亮线
    before = _clone_faces(faces)

    fixed = seam_fix(faces)

    assert faces == before
    assert len(fixed) == 6
    assert all(len(face) == 8 and all(len(row) == 8 for row in face) for face in fixed)
    values = [value for face in fixed for row in face for value in row]
    assert max(values) < 0.5  # 亮线被接缝两侧的平均值消掉
    assert max(values) > 0.2 + TOL  # 不是简单压暗
    interior = [
        fixed[face][row][col]
        for face in range(6)
        for row in range(1, 7)
        for col in range(1, 7)
    ]
    assert all(value == pytest.approx(0.2, abs=TOL) for value in interior)

    uniform = seam_fix(_faces(6, 0.35))
    assert all(
        value == pytest.approx(0.35, abs=TOL)
        for face in uniform
        for row in face
        for value in row
    )


# 缺陷 7：面数不为 6 不报错，缺面全黑。
def test_validate_flags_wrong_face_count():
    five = _faces(4)[:5]
    problems = validate_cubemap(five, 4)
    assert problems
    assert any("face" in message.lower() for message in problems)

    seven = _faces(4) + [_face(4, 0.0)]
    problems = validate_cubemap(seven, 4)
    assert problems
    assert any("face" in message.lower() for message in problems)

    empty_face = _faces(4)
    empty_face[2] = []
    problems = validate_cubemap(empty_face, 4)
    assert problems
    assert any("face" in message.lower() for message in problems)

    assert validate_cubemap(_faces(4), 4) == []


# 缺陷 8：尺寸不等不校验，越界。
def test_validate_flags_size_mismatch_and_nan():
    faces = _faces(4)
    faces[5] = _face(5, 0.2)
    problems = validate_cubemap(faces, 4)
    assert problems
    assert any("size" in message.lower() for message in problems)

    nan_faces = _faces(4)
    nan_faces[2][1][2] = float("nan")
    problems = validate_cubemap(nan_faces, 4)
    assert problems
    assert any(
        "nan" in message.lower() or "finite" in message.lower()
        for message in problems
    )

    with pytest.raises(CubemapError):
        prefilter_batch([{"faces": faces}], {"roughness": 0.5, "samples": 8})


# 缺陷 9：批量共用输出面，批次之间串。
def test_batch_outputs_are_isolated_between_calls():
    cubemap = _cubemap(6, 0.3)
    before = _clone_faces(cubemap["faces"])
    config = {"roughness": 0.5, "samples": 8}

    first = prefilter_batch([cubemap], config)
    frozen = _clone_result(first)
    first["cubemaps"][0]["faces"][0][0][0] = 9.0

    second = prefilter_batch([cubemap], config)

    _assert_result_close(second, frozen)
    assert second["cubemaps"][0] is not first["cubemaps"][0]
    assert second["cubemaps"][0]["faces"][0] is not first["cubemaps"][0]["faces"][0]
    assert first["cubemaps"][0]["faces"][0][0][0] == 9.0
    assert cubemap["faces"] == before


# 缺陷 10：samples=0 不校验，除零。
def test_prefilter_rejects_zero_samples():
    face = _face(8, 0.5)
    with pytest.raises(CubemapError):
        prefilter_face(face, 0.5, 0)
    with pytest.raises(CubemapError):
        prefilter_face(face, 0.5, -2)


# 组合 1：roughness=0 时输出接近原面（1e-6）。
def test_zero_roughness_preserves_face():
    face = _gradient_face(8)
    result = prefilter_face(face, 0.0, 16)

    assert len(result) == len(face)
    _assert_face_close(result, face)


# 组合 2：+Z 方向映射到面 4。
def test_positive_z_maps_to_face_four():
    face, u, v = dir_to_face_uv((0.0, 0.0, 1.0))

    assert face == 4
    assert u == pytest.approx(0.5, abs=TOL)
    assert v == pytest.approx(0.5, abs=TOL)


# 组合 3：空面抛 CubemapError。
def test_empty_face_raises():
    with pytest.raises(CubemapError):
        prefilter_face([], 0.5, 8)
    with pytest.raises(CubemapError):
        prefilter_face([[]], 0.5, 8)


# 组合 4：samples=0 抛 CubemapError（批量入口）。
def test_batch_rejects_zero_samples():
    with pytest.raises(CubemapError):
        prefilter_batch([_cubemap(4, 0.4)], {"roughness": 0.5, "samples": 0})


# 组合 5：批量全成功时顺序不变。
def test_batch_success_preserves_order():
    cubemaps = [_cubemap(4, value) for value in (0.1, 0.4, 0.7)]
    config = {"roughness": 0.5, "samples": 8}

    result = prefilter_batch(cubemaps, config)

    assert result["count"] == 3
    assert len(result["cubemaps"]) == 3
    for got, source in zip(result["cubemaps"], cubemaps):
        expected = [prefilter_face(face, 0.5, 8) for face in source["faces"]]
        for got_face, want_face in zip(got["faces"], expected):
            _assert_face_close(got_face, want_face)
    first = result["cubemaps"][0]["faces"][0][0][0]
    second = result["cubemaps"][1]["faces"][0][0][0]
    third = result["cubemaps"][2]["faces"][0][0][0]
    assert first < second < third
