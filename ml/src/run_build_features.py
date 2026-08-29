"""Entry point AN TOÀN để build feature artifact — xem ghi chú ở cuối
ml/src/features/build.py. File này CHỈ import main() (không định nghĩa class
nào), nên build.py luôn được nạp như module bình thường, không phải __main__.
"""
from ml.src.features.build import main

if __name__ == "__main__":
    main()
