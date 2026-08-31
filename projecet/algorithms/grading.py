import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 常用公路集料筛孔(mm)

SIEVE_SIZE = np.array(
    [
        53.0,
        37.5,
        31.5,
        26.5,
        19,
        16.0,
        9.5,
        4.75,
    ]
)
# SIEVE_SIZE = np.array(
#     [
#         26.5,
#         19,
#         13.2,
#         9.5,
#         4.75,
#         2.36,
#         1.18,
#         0.6,
#         0.3,
#         0.15
#     ]
# )


# 标准级配范围（GB/T 14685-2022 表1，连续粒级，累计筛余换算为累计通过率）
# 每条记录：(筛孔mm, 通过率下限%, 通过率上限%)
SPEC_RANGES = {
    "5~16 连续级配": [
        (2.36, 0, 5), (4.75, 0, 15), (9.5, 40, 70), (16.0, 90, 100), (19.0, 100, 100),
    ],
    "5~20 连续级配": [
        (2.36, 0, 5), (4.75, 0, 10), (9.5, 20, 60), (19.0, 90, 100), (26.5, 100, 100),
    ],
    "5~25 连续级配": [
        (2.36, 0, 5), (4.75, 0, 10), (16.0, 30, 70), (26.5, 95, 100), (31.5, 100, 100),
    ],
    "5~31.5 连续级配": [
        (2.36, 0, 5), (4.75, 0, 10), (9.5, 10, 30), (19.0, 55, 85), (31.5, 95, 100),
    ],
    "5~40 连续级配": [
        (2.36, 0, 5), (4.75, 0, 5), (16.0, 35, 70), (31.5, 90, 100), (37.5, 100, 100),
    ],
}



def calculate_grading(
        particles
):

    """
    输入:
        particles:
        [
        {
        diameter_mm: xx
        }
        ]

    输出:
        dataframe，每行对应一个筛孔，包含:
        分计筛余(%)、累计筛余(%)、累计通过率(%)、数量占比(%)
    """

    diameters = np.array(
        [
            p["diameter_mm"]
            for p in particles
        ]
    )

    if len(diameters) == 0:
        return None

    # 用面积近似质量（后期可替换密度修正）
    weights = diameters ** 2
    total_weight = np.sum(weights)
    total_count = len(diameters)

    result = []
    prev_retain_w = 0.0
    prev_retain_n = 0
    for sieve in SIEVE_SIZE:
        # 大于当前筛孔的颗粒 = 累计筛余在该筛上的部分
        retain_w = weights[diameters > sieve].sum()
        retain_n = int((diameters > sieve).sum())
        # 本档分计筛余 = 相邻两级累计筛余之差（即 (sieve, 上一筛孔] 区间）
        fraction_w = retain_w - prev_retain_w
        fraction_n = retain_n - prev_retain_n
        result.append(
            [
                sieve,
                fraction_w / total_weight * 100,
                retain_w / total_weight * 100,
                100 - retain_w / total_weight * 100,
                fraction_n / total_count * 100,
            ]
        )
        prev_retain_w = retain_w
        prev_retain_n = retain_n

    df = pd.DataFrame(
        result,
        columns=[
            "筛孔(mm)",
            "分计筛余(%)",
            "累计筛余(%)",
            "累计通过率(%)",
            "数量占比(%)",
        ]
    )

    return df

