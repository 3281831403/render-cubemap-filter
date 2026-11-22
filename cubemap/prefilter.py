"""立方体贴图预滤波。

约定
----
* 立方体贴图是 ``{"faces": [face_0, ..., face_5]}``，面顺序固定为
  ``0=+X、1=-X、2=+Y、3=-Y、4=+Z、5=-Z``，因此 ``+Z`` 方向属于面 4。
* 每个面是 ``n × n`` 的二维列表 ``face[row][col]``，纹素是有限浮点
  辐照度；``row`` 从上到下，``col`` 从左到右，纹素中心为
  ``u = (col + 0.5) / n``、``v = (row + 0.5) / n``。
* 面方向按立方体贴图约定重建（方向先归一化）：
  ``+X = (1, -(2v-1), -(2u-1))``、``-X = (-1, -(2v-1), (2u-1))``、
  ``+Y = (2u-1, 1, 2v-1)``、``-Y = (2u-1, -1, -(2v-1))``、
  ``+Z = (2u-1, -(2v-1), 1)``、``-Z = (-(2u-1), -(2v-1), -1)``。
* ``dir_to_face_uv`` 接受任意非零方向向量，内部归一化后再选面；``u``、
  ``v`` 落在 ``[0.0, 1.0]``。
* 粗糙度 ``roughness`` 取 ``[0.0, 1.0]``：``0.0`` 为镜面，``1.0`` 最粗糙。
* 浮点比较容差为 ``1e-6``；采样序列完全确定，不得使用随机模块，保证
  离线、可复现。
* 非法输入统一抛出 :class:`CubemapError`。
"""

from __future__ import annotations

import math

FACE_COUNT = 6
FACE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
TOLERANCE = 1e-6
GOLDEN_RATIO = 0.6180339887498949


class CubemapError(ValueError):
    """立方体贴图输入、配置或批量状态非法。"""


def _is_number(value: object) -> bool:
    """判断是否是实数（排除 bool）。"""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _check_roughness(roughness: object) -> float:
    if not _is_finite_number(roughness):
        raise CubemapError("roughness 必须是有限实数")
    value = float(roughness)
    if value < 0.0 or value > 1.0:
        raise CubemapError("roughness 必须位于 [0.0, 1.0]")
    return value


def _check_samples(samples: object) -> int:
    if not isinstance(samples, int) or isinstance(samples, bool):
        raise CubemapError("samples 必须是正整数")
    if samples <= 0:
        raise CubemapError("samples 必须大于 0")
    return samples


def _check_positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CubemapError(f"{name} 必须是正整数")
    return value


def _normalize3(
    vector: object, name: str = "方向"
) -> tuple[float, float, float]:
    """把三分量向量归一化，非法时抛 :class:`CubemapError`。"""

    if not isinstance(vector, (tuple, list)) or len(vector) != 3:
        raise CubemapError(f"{name} 必须是三个分量的向量")
    if not all(_is_finite_number(component) for component in vector):
        raise CubemapError(f"{name} 的分量必须是有限实数")
    x, y, z = (float(component) for component in vector)

    # 先按最大分量缩放，避免平方溢出/下溢。
    scale = max(abs(x), abs(y), abs(z))
    if scale == 0.0:
        raise CubemapError(f"{name} 不能是零向量")
    x, y, z = x / scale, y / scale, z / scale
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0.0 or not math.isfinite(length):
        raise CubemapError(f"{name} 不能是零向量")
    return (x / length, y / length, z / length)


def _direction_from_face_uv(face: int, u: float, v: float) -> tuple[float, float, float]:
    """按面内坐标重建（未归一化的）方向。"""

    su = 2.0 * u - 1.0
    sv = 2.0 * v - 1.0
    if face == 0:  # +X
        return (1.0, -sv, -su)
    if face == 1:  # -X
        return (-1.0, -sv, su)
    if face == 2:  # +Y
        return (su, 1.0, sv)
    if face == 3:  # -Y
        return (su, -1.0, -sv)
    if face == 4:  # +Z
        return (su, -sv, 1.0)
    if face == 5:  # -Z
        return (-su, -sv, -1.0)
    raise CubemapError(f"未知的面索引 {face!r}")


def dir_to_face_uv(direction: tuple[float, float, float]) -> tuple[int, float, float]:
    """把方向向量映射为 ``(face, u, v)``。

    参数
    ----
    direction:
        三分量方向向量，必须是非零有限实数。调用者不需要预先归一化：
        函数必须先归一化方向，再用绝对值最大的主轴决定面，因此
        ``(0, 0, 0.01)`` 与 ``(0, 0, 100)`` 必须得到同一个面、同一组
        ``u``、``v``。主轴并列时按 ``+X/-X`` → ``+Y/-Y`` → ``+Z/-Z``
        的顺序优先。

    返回
    ----
    ``(face, u, v)``：``face`` 为 ``0..5``（``0=+X、1=-X、2=+Y、
    3=-Y、4=+Z、5=-Z``），``u``、``v`` 是面内坐标，均位于
    ``[0.0, 1.0]``，按 ``1e-6`` 容差比较。``+Z`` 方向必须落在面 4。

    异常
    ----
    CubemapError: 方向不是三个有限实数，或为零向量。
    """

    x, y, z = _normalize3(direction)
    ax, ay, az = abs(x), abs(y), abs(z)

    # 主轴并列时按 X → Y → Z 优先。
    if ax >= ay and ax >= az:
        if x > 0.0:
            face = 0
            u = 0.5 * (1.0 - z / x)
            v = 0.5 * (1.0 - y / x)
        else:
            face = 1
            u = 0.5 * (1.0 - z / x)
            v = 0.5 * (1.0 + y / x)
    elif ay >= az:
        if y > 0.0:
            face = 2
            u = 0.5 * (1.0 + x / y)
            v = 0.5 * (1.0 + z / y)
        else:
            face = 3
            u = 0.5 * (1.0 - x / y)
            v = 0.5 * (1.0 + z / y)
    else:
        if z > 0.0:
            face = 4
            u = 0.5 * (1.0 + x / z)
            v = 0.5 * (1.0 - y / z)
        else:
            face = 5
            u = 0.5 * (1.0 + x / z)
            v = 0.5 * (1.0 + y / z)

    # 数值噪声可能让结果略微越过边界。
    u = min(max(u, 0.0), 1.0)
    v = min(max(v, 0.0), 1.0)
    return face, u, v


def texel_angle(face_size: int) -> float:
    """返回单个纹素对应的平均立体角（球面度），用于 mip 选择。

    立方体贴图覆盖整个球面，总立体角为 ``4π``；它由 6 个面均分，每个面
    再均分给 ``face_size²`` 个纹素。因此本函数返回
    ``4π / (6 * face_size²)``，也就是 ``2π / (3 * face_size²)``。
    不能把面投影到平面上、用平面像素面积 ``(2 / face_size)²`` 代替立体
    角，那会高估纹素大小、导致选出的 mip 偏小（画面偏锐）。

    参数
    ----
    face_size:
        每个面的边长（纹素数），必须是正整数。

    返回
    ----
    平均单纹素立体角，按 ``1e-6`` 容差比较；``6 * face_size² *
    texel_angle(face_size)`` 必须等于 ``4π``。

    异常
    ----
    CubemapError: ``face_size`` 不是正整数。
    """

    size = _check_positive_int(face_size, "face_size")
    # 4π 均分给 6 个面，每个面再均分给 size² 个纹素。
    return 4.0 * math.pi / (FACE_COUNT * size * size)


def mip_for_roughness(roughness: float, max_mip: int) -> int:
    """把粗糙度映射为 mip 级别，结果向上取整。

    连续级别取 ``roughness * max_mip``，返回值是其向上取整并夹紧到
    ``[0, max_mip]``：``roughness == 0`` 返回 ``0``，``roughness == 1``
    返回 ``max_mip``。必须向上取整，向下取整会让粗糙面停留在过小的
    mip 上、依旧锐利。例如 ``max_mip == 6`` 时 ``roughness == 0.3``
    的连续级别是 ``1.8``，应返回 ``2``。

    参数
    ----
    roughness:
        粗糙度，必须是 ``[0.0, 1.0]`` 内的有限实数。
    max_mip:
        最大 mip 级别，必须是非负整数。

    返回
    ----
    ``[0, max_mip]`` 内的整数，且随 ``roughness`` 单调不减。

    异常
    ----
    CubemapError: 粗糙度越界或 ``max_mip`` 非法。
    """

    value = _check_roughness(roughness)
    if not isinstance(max_mip, int) or isinstance(max_mip, bool) or max_mip < 0:
        raise CubemapError("max_mip 必须是非负整数")

    level = math.ceil(value * max_mip)
    # 浮点误差可能让 ceil 越界，夹紧到合法区间。
    if level < 0:
        level = 0
    if level > max_mip:
        level = max_mip
    return level


def ggx_sample(
    normal: tuple[float, float, float],
    roughness: float,
    xi: tuple[float, float],
) -> tuple[float, float, float]:
    """按 GGX 分布重要性采样一个单位半程向量。

    采样必须随粗糙度变化：``roughness == 0`` 时直接返回归一化后的
    ``normal``（镜面反射）；``roughness`` 越大，半程向量偏离 ``normal``
    越远。使用两个确定性样本 ``xi = (u1, u2)``，不得调用随机模块：

    * ``a = roughness²``；
    * ``cosθ = sqrt((1 - u1) / (1 + (a² - 1) * u1))``；
    * ``φ = 2π * u2``，在以 ``normal`` 为 Z 轴的正交基里合成方向。

    参数
    ----
    normal:
        非零有限三分量法线，不需要预先归一化。
    roughness:
        粗糙度，取值 ``[0.0, 1.0]``。
    xi:
        两个分量均位于 ``[0.0, 1.0)`` 的确定性样本；相同输入必须得到
        逐分量相同的输出。

    返回
    ----
    长度为 3 的单位向量，且位于 ``normal`` 所在半球
    （``dot(result, normal) >= 0``），按 ``1e-6`` 容差比较。

    异常
    ----
    CubemapError: 法线、粗糙度或 ``xi`` 非法。
    """

    nx, ny, nz = _normalize3(normal, "normal")
    value = _check_roughness(roughness)

    if not isinstance(xi, (tuple, list)) or len(xi) != 2:
        raise CubemapError("xi 必须是两个分量的样本")
    if not all(_is_finite_number(component) for component in xi):
        raise CubemapError("xi 的分量必须是有限实数")
    u1, u2 = float(xi[0]), float(xi[1])
    if not (0.0 <= u1 < 1.0 and 0.0 <= u2 < 1.0):
        raise CubemapError("xi 的分量必须位于 [0.0, 1.0)")

    if value == 0.0:
        # 镜面：半程向量就是法线本身。
        return (nx, ny, nz)

    a = value * value
    denominator = 1.0 + (a * a - 1.0) * u1
    cos_theta = math.sqrt(max((1.0 - u1) / denominator, 0.0))
    cos_theta = min(max(cos_theta, -1.0), 1.0)
    sin_theta = math.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    phi = 2.0 * math.pi * u2

    # 以 normal 为 Z 轴构造正交基；选择与 normal 不平行的辅助轴。
    if abs(nz) < 0.999:
        helper = (0.0, 0.0, 1.0)
    else:
        helper = (1.0, 0.0, 0.0)
    tangent = _cross(helper, (nx, ny, nz))
    tangent = _normalize_vec(tangent)
    bitangent = _cross((nx, ny, nz), tangent)

    tx = sin_theta * math.cos(phi)
    ty = sin_theta * math.sin(phi)
    tz = cos_theta
    result = (
        tangent[0] * tx + bitangent[0] * ty + nx * tz,
        tangent[1] * tx + bitangent[1] * ty + ny * tz,
        tangent[2] * tx + bitangent[2] * ty + nz * tz,
    )
    return _normalize_vec(result)


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize_vec(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(
        vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]
    )
    if length == 0.0:
        return vector
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _frac(value: float) -> float:
    return value - math.floor(value)


def _validate_face_matrix(face: object) -> int:
    """检查一个面是合法的非空方阵，返回边长。"""

    if not isinstance(face, list):
        raise CubemapError("面必须是二维列表")
    if len(face) == 0:
        raise CubemapError("面不能为空")
    size = len(face)
    for row_index, row in enumerate(face):
        if not isinstance(row, (list, tuple)):
            raise CubemapError(f"面的第 {row_index} 行不是列表")
        if len(row) != size:
            raise CubemapError("面必须是 n×n 方阵")
        for col_index, value in enumerate(row):
            if not _is_finite_number(value):
                raise CubemapError(
                    f"纹素 ({row_index}, {col_index}) 必须是有限实数"
                )
    return size


def _sample_face_bilinear(face: list[list[float]], u: float, v: float) -> float:
    """在面内做 clamp-to-edge 双线性采样。"""

    size = len(face)
    max_index = size - 1
    x = u * size - 0.5
    y = v * size - 0.5
    x = min(max(x, 0.0), float(max_index))
    y = min(max(y, 0.0), float(max_index))

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, max_index)
    y1 = min(y0 + 1, max_index)
    fx = x - x0
    fy = y - y0

    top = face[y0][x0] * (1.0 - fx) + face[y0][x1] * fx
    bottom = face[y1][x0] * (1.0 - fx) + face[y1][x1] * fx
    return top * (1.0 - fy) + bottom * fy


def prefilter_face(
    face: list[list[float]],
    roughness: float,
    samples: int,
) -> list[list[float]]:
    """对一个面逐纹素做预滤波，输出与输入同尺寸。

    对每个纹素中心 ``(u, v)``，用确定性序列
    ``xi_s = ((s + 0.5) / samples, frac((s + 1) * GOLDEN_RATIO))``
    调用 :func:`ggx_sample`（局部法线取 ``(0, 0, 1)``），把半程向量的
    切向分量乘以 ``0.5`` 当作 UV 偏移，在输入面上做 clamp-to-edge 的
    双线性采样，最后对 ``samples`` 个样本取算术平均。必须除以样本数，
    不能只把样本相加，否则亮度会随 ``samples`` 变化。

    ``roughness == 0`` 时所有偏移为零，输出必须与输入逐纹素一致
    （``1e-6`` 容差）。

    参数
    ----
    face:
        非空 ``n × n`` 二维列表，纹素为有限浮点；``n >= 1``。
    roughness:
        粗糙度，取值 ``[0.0, 1.0]``。
    samples:
        每纹素采样数，必须是正整数；``samples <= 0`` 必须在任何除法
        之前抛出 :class:`CubemapError`，不能产生 ``ZeroDivisionError``。

    返回
    ----
    新的 ``n × n`` 二维列表，不修改输入；每个纹素是 ``samples`` 个
    样本的算术平均，按 ``1e-6`` 容差比较。

    异常
    ----
    CubemapError: 面结构、粗糙度或采样数非法。
    """

    value = _check_roughness(roughness)
    count = _check_samples(samples)
    size = _validate_face_matrix(face)

    result = [[0.0 for _ in range(size)] for _ in range(size)]

    if value == 0.0:
        # 镜面：偏移全为零，直接复制（独立分配）。
        for row in range(size):
            for col in range(size):
                result[row][col] = float(face[row][col])
        return result

    normal = (0.0, 0.0, 1.0)
    for row in range(size):
        v = (row + 0.5) / size
        for col in range(size):
            u = (col + 0.5) / size
            total = 0.0
            for sample_index in range(count):
                xi = (
                    (sample_index + 0.5) / count,
                    _frac((sample_index + 1) * GOLDEN_RATIO),
                )
                half_vector = ggx_sample(normal, value, xi)
                # 切向分量乘以 0.5 作为 UV 偏移。
                total += _sample_face_bilinear(
                    face,
                    u + half_vector[0] * 0.5,
                    v + half_vector[1] * 0.5,
                )
            # 按样本数取平均，亮度与采样数无关。
            result[row][col] = total / count
    return result


def _collect_problems(faces: object, size: object) -> list[str]:
    """收集立方体贴图的问题；``size`` 为 ``None`` 时只检查自洽性。"""

    problems: list[str] = []

    expected: int | None = None
    if size is None:
        pass
    elif not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        problems.append("size 必须是正整数")
    else:
        expected = size

    if not isinstance(faces, list):
        problems.append("faces 必须是包含 6 个面的列表")
        return problems

    if len(faces) != FACE_COUNT:
        problems.append(
            f"faces 必须恰好包含 {FACE_COUNT} 个面，实际为 {len(faces)} 个"
        )

    sizes: list[int] = []
    for face_index, face in enumerate(faces):
        name = FACE_NAMES[face_index] if face_index < FACE_COUNT else str(face_index)
        if not isinstance(face, list):
            problems.append(f"face {face_index} ({name}) 必须是二维列表")
            continue
        if len(face) == 0:
            problems.append(f"face {face_index} ({name}) 不能为空")
            continue
        rows = len(face)
        sizes.append(rows)
        if expected is not None and rows != expected:
            problems.append(
                f"face {face_index} ({name}) size 为 {rows}，期望 {expected}"
            )
        for row_index, row in enumerate(face):
            if not isinstance(row, (list, tuple)):
                problems.append(
                    f"face {face_index} ({name}) 第 {row_index} 行不是列表"
                )
                continue
            if len(row) != rows:
                problems.append(
                    f"face {face_index} ({name}) size 不一致："
                    f"第 {row_index} 行有 {len(row)} 列，期望 {rows}"
                )
            for col_index, value in enumerate(row):
                if not _is_number(value):
                    problems.append(
                        f"face {face_index} ({name}) 纹素 "
                        f"({row_index}, {col_index}) 不是有限实数"
                    )
                elif math.isnan(float(value)):
                    problems.append(
                        f"face {face_index} ({name}) 纹素 "
                        f"({row_index}, {col_index}) 是 nan"
                    )
                elif not math.isfinite(float(value)):
                    problems.append(
                        f"face {face_index} ({name}) 纹素 "
                        f"({row_index}, {col_index}) 不是有限数 (non-finite)"
                    )

    if len(set(sizes)) > 1:
        problems.append(f"各面 size 不一致: {sizes}")

    return problems


def seam_fix(faces: list[list[list[float]]]) -> list[list[list[float]]]:
    """对齐立方体贴图的接缝，消除边缘亮线。

    对每个边界纹素，把它投影到立方体的棱上：列 ``0`` / ``n-1`` 分别
    取 ``u = 0.0`` / ``u = 1.0``，行 ``0`` / ``n-1`` 分别取 ``v = 0.0`` /
    ``v = 1.0``，其余坐标仍取纹素中心；角纹素同时取到角点。棱点（或角
    点）方向相同（按 ``1e-9`` 量化）的边界纹素属于同一接缝组，组内取
    算术平均后写回，于是接缝两侧数值一致、不会出现亮线。内部纹素原样
    保留。

    参数
    ----
    faces:
        恰好 6 个等大 ``n × n`` 面的列表，每个纹素是有限浮点；
        ``n >= 1``。

    返回
    ----
    与输入同尺寸的 6 个新面，不修改输入；常量贴图必须原样返回，按
    ``1e-6`` 容差比较。

    异常
    ----
    CubemapError: 面数不为 6、尺寸不等或纹素非法。
    """

    if not isinstance(faces, list):
        raise CubemapError("faces 必须是包含 6 个面的列表")
    if len(faces) != FACE_COUNT:
        raise CubemapError(f"faces 必须恰好包含 {FACE_COUNT} 个面")
    sizes = [_validate_face_matrix(face) for face in faces]
    if len(set(sizes)) != 1:
        raise CubemapError(f"各面尺寸不一致: {sizes}")
    size = sizes[0]

    result = [[row[:] for row in face] for face in faces]

    # 按 1e-9 量化后的棱/角方向分组。
    groups: dict[tuple[float, float, float], list[tuple[int, int, int]]] = {}
    for face_index in range(FACE_COUNT):
        for row in range(size):
            for col in range(size):
                on_boundary = (
                    row == 0
                    or row == size - 1
                    or col == 0
                    or col == size - 1
                )
                if not on_boundary:
                    continue
                if col == 0:
                    u = 0.0
                elif col == size - 1:
                    u = 1.0
                else:
                    u = (col + 0.5) / size
                if row == 0:
                    v = 0.0
                elif row == size - 1:
                    v = 1.0
                else:
                    v = (row + 0.5) / size

                direction = _normalize_vec(
                    _direction_from_face_uv(face_index, u, v)
                )
                key = (
                    round(direction[0], 9),
                    round(direction[1], 9),
                    round(direction[2], 9),
                )
                groups.setdefault(key, []).append((face_index, row, col))

    for members in groups.values():
        if len(members) < 2:
            continue
        average = (
            sum(faces[face_index][row][col] for face_index, row, col in members)
            / len(members)
        )
        for face_index, row, col in members:
            result[face_index][row][col] = average

    return result


def validate_cubemap(faces: list[list[list[float]]], size: int) -> list[str]:
    """检查立方体贴图，返回问题描述列表；合法时返回空列表。

    至少检查以下问题：

    * ``faces`` 不是列表，或面数不为 6（缺面会让对应方向直接全黑）；
    * 任一面不是方阵，或边长不等于 ``size``，或各面尺寸不一致
      （会导致采样越界）；
    * 任一纹素不是有限实数，包括 ``NaN`` 与 ``inf``。

    返回的每条消息应包含对应字段名（如 ``"faces"``、``"size"``、
    ``"nan"``、``"finite"``），便于调用者定位。本函数只收集问题，不
    抛异常；即使输入畸形也返回问题列表。

    参数
    ----
    faces:
        待检查的面列表。
    size:
        期望的每个面边长，必须是正整数。

    返回
    ----
    问题描述字符串列表，合法时为 ``[]``。

    异常
    ----
    无。
    """

    try:
        return _collect_problems(faces, size)
    except Exception as error:  # 本函数承诺不抛异常
        return [f"faces 校验失败: {error}"]


def prefilter_batch(cubemaps: list[dict], config: dict) -> dict:
    """批量做立方体贴图预滤波，任一输入非法时整体回滚。

    参数
    ----
    cubemaps:
        ``{"faces": [...]}`` 列表，每个元素必须先通过
        :func:`validate_cubemap`。
    config:
        至少包含 ``"roughness"`` 与 ``"samples"``，所有贴图共用同一
        配置；``samples <= 0`` 必须抛 :class:`CubemapError`。

    返回
    ----
    ``{"cubemaps": [...], "count": n}``：``cubemaps`` 中第 ``i`` 个元素
    是第 ``i`` 个输入的预滤波结果 ``{"faces": [...]}``，顺序与输入完全
    一致；``count == len(cubemaps)``。空输入返回空列表和 ``count == 0``。

    原子性
    ------
    必须先校验全部输入与配置，再计算并组装返回值。任一贴图或配置非法
    都必须抛出 :class:`CubemapError`，不得返回部分结果、不得原地修改
    输入。每个输出面都要独立分配，不能与输入或上一次批量调用共享，
    否则批次之间会互相串扰。

    异常
    ----
    CubemapError: 配置非法或任一贴图非法。
    """

    if not isinstance(config, dict):
        raise CubemapError("config 必须是字典")
    if "roughness" not in config:
        raise CubemapError("config 缺少 roughness")
    if "samples" not in config:
        raise CubemapError("config 缺少 samples")
    roughness = _check_roughness(config["roughness"])
    samples = _check_samples(config["samples"])

    if not isinstance(cubemaps, list):
        raise CubemapError("cubemaps 必须是列表")

    # 先整体校验：任一非法立即回滚，不产生部分结果。
    validated: list[list[list[float]]] = []
    for index, cubemap in enumerate(cubemaps):
        if not isinstance(cubemap, dict):
            raise CubemapError(f"cubemap[{index}] 必须是字典")
        if "faces" not in cubemap:
            raise CubemapError(f"cubemap[{index}] 缺少 faces")
        faces = cubemap["faces"]
        problems = _collect_problems(faces, None)
        if problems:
            raise CubemapError(
                f"cubemap[{index}] 非法: " + "; ".join(problems)
            )
        validated.append(faces)

    # 校验全部通过后再计算，输出逐项独立分配。
    results = []
    for faces in validated:
        output_faces = [
            prefilter_face(face, roughness, samples) for face in faces
        ]
        results.append({"faces": output_faces})

    return {"cubemaps": results, "count": len(results)}
