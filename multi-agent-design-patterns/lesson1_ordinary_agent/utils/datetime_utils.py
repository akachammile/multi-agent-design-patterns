"""
日期时间工具函数模块，用于一致的时区处理。

后端以 UTC 格式存储时间戳，并以带有明确时区标识符的 ISO 8601 字符串格式对外暴露。
在面向用户的显示场景中，通常会转换为 Asia/Shanghai（上海时区）。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from zoneinfo import ZoneInfo

UTC = dt.UTC
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
_ISO_Z_SUFFIX = "+00:00"


def utc_now() -> dt.datetime:
    """返回当前 UTC 时间，包含时区信息的 datetime 对象。"""
    return dt.datetime.now(UTC)


def utc_now_naive() -> dt.datetime:
    """返回当前 UTC 时间，不包含时区信息的 datetime 对象（用于兼容旧版数据库字段）。"""
    return dt.datetime.now(UTC).replace(tzinfo=None)


def shanghai_now() -> dt.datetime:
    """返回当前上海时区时间，包含时区信息的 datetime 对象。"""
    return utc_now().astimezone(SHANGHAI_TZ)


def ensure_utc(value: dt.datetime) -> dt.datetime:
    """
    将 datetime 转换为 UTC 时区。

    无时区信息的值将被假定为上海时区，以兼容历史数据。
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=SHANGHAI_TZ)
    return value.astimezone(UTC)


def ensure_shanghai(value: dt.datetime) -> dt.datetime:
    """
    将 datetime 转换为上海时区。

    无时区信息的值将被假定为上海时区（历史兼容行为）。
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=SHANGHAI_TZ)
    return value.astimezone(SHANGHAI_TZ)


def utc_isoformat(value: dt.datetime | None = None) -> str:
    """返回 UTC 时区的 ISO 8601 格式字符串，末尾带 Z 后缀。"""
    value = ensure_utc(value or utc_now())
    iso_string = value.isoformat()
    if iso_string.endswith(_ISO_Z_SUFFIX):
        return iso_string.replace(_ISO_Z_SUFFIX, "Z")
    return iso_string


def shanghai_isoformat(value: dt.datetime | None = None) -> str:
    """返回上海时区的 ISO 8601 格式字符串。"""
    value = ensure_shanghai(value or shanghai_now())
    return value.isoformat()


def coerce_datetime(value: dt.datetime | None) -> dt.datetime | None:
    """将持久化的 datetime 规范化为 UTC 时区，优雅地处理空值。"""
    if value is None:
        return None
    return ensure_utc(value)


def coerce_any_to_utc_datetime(
    value: dt.datetime | int | float | str | None,
) -> dt.datetime | None:
    """
    将各种时间戳格式转换为包含时区信息的 UTC datetime。

    支持的格式：
      * 包含或不包含时区信息的 datetime 对象
      * Unix 时间戳（秒），支持 int/float 类型
      * ISO 8601 格式字符串
    """
    if value is None:
        return None

    if isinstance(value, dt.datetime):
        return ensure_utc(value)

    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value, tz=UTC)

    if isinstance(value, str):
        # 尝试解析 ISO 8601 格式字符串
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", _ISO_Z_SUFFIX))
            return ensure_utc(parsed)
        except ValueError:
            # 尝试回退到数字字符串解析
            try:
                as_number = float(value)
                return dt.datetime.fromtimestamp(as_number, tz=UTC)
            except ValueError:
                raise ValueError(f"不支持的日期时间字符串格式: {value!r}") from None

    raise TypeError(f"不支持的日期时间值: {value!r}")


def normalize_iterable_to_utc(
    values: Iterable[dt.datetime | None],
) -> list[dt.datetime | None]:
    """将可迭代对象中的每个 datetime 规范化为 UTC 时区。"""
    return [
        coerce_datetime(item) if isinstance(item, dt.datetime) else None
        for item in values
    ]


def format_utc_datetime(value: dt.datetime | None) -> str | None:
    """
    将 datetime 格式化为 UTC 时区的 ISO 8601 字符串，处理无时区信息的 datetime。

    输入为 None 时返回 None。
    无时区信息的 datetime 将被假定为 UTC 时区（历史兼容行为）。
    """
    if value is None:
        return None
    return utc_isoformat(value)


__all__ = [
    "UTC",
    "SHANGHAI_TZ",
    "utc_now",
    "utc_now_naive",
    "shanghai_now",
    "ensure_utc",
    "ensure_shanghai",
    "utc_isoformat",
    "shanghai_isoformat",
    "coerce_datetime",
    "coerce_any_to_utc_datetime",
    "normalize_iterable_to_utc",
    "format_utc_datetime",
]
