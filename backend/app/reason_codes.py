"""Re-export thuần từ ml.src.reason_codes — xem ghi chú biên giới tầng trong scorer.py.

File này tồn tại (thay vì import thẳng ml.src.reason_codes ở nơi dùng) để khớp
cấu trúc thư mục mục tiêu trong docs/NOTES.md và làm rõ: đây LÀ điểm phụ thuộc
tường minh của backend vào ml/, không phải import rải rác khó theo dõi.
"""
from ml.src.reason_codes import build_reason_codes, direction_phrase, humanize

__all__ = ["build_reason_codes", "direction_phrase", "humanize"]
