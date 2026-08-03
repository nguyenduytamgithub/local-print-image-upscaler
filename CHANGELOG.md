# Changelog

Các thay đổi đáng chú ý của Local Print Image Upscaler được ghi tại đây. Dự án
dùng [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-08-03

### Added

- Một lệnh `upscale` thống nhất cho V2 Fast và V3 High.
- Xử lý một ảnh hoặc toàn bộ ảnh trong một thư mục và các thư mục con.
- Giữ cấu trúc thư mục con, tránh trùng tên và tiếp tục khi một ảnh bị hỏng.
- Giới hạn kích thước, khóa GPU và thay file kết quả theo cơ chế an toàn.
- Manifest kỹ thuật có hash nguồn, hash kết quả và phiên bản ứng dụng.
- Kiểm thử padding/tile cho cả ảnh rất nhỏ và ảnh có một chiều chỉ một pixel.

### Packaging

- Tách ảnh người dùng, output, model, runtime, QA và file tạm khỏi source Git.
- Bổ sung README công khai, thông báo giấy phép bên thứ ba và khóa phiên bản.
- Phát hành phần code riêng của dự án theo MIT License.
