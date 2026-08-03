# Changelog

Các thay đổi đáng chú ý của Local Print Image Upscaler được ghi tại đây. Dự án dùng
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-08-04

### Added

- Thêm `upscale layers` (V5 Smart Layers) để suy luận một số lượng hữu hạn layer raster hữu ích từ
  PNG/JPEG/TIFF/WebP/BMP phẳng; hỗ trợ một ảnh hoặc batch cả thư mục, cùng hệ số `n=1..20`.
- Phân luồng poster/đồ họa và ảnh tự nhiên: SAM 2.1 sinh đề xuất mask; poster được gộp theo panel,
  hàng, quan hệ cha–con và hình học, còn ảnh nhiều texture dùng chiến lược nhóm bảo thủ riêng.
- Thêm Grounding DINO làm nhãn gợi ý và Tesseract với model `tessdata_best` tiếng Việt đã pin để lấy
  vùng/dòng OCR. Các tín hiệu này không biến chữ bitmap thành font có thể gõ sửa.
- Tạo nền dưới vật thể bằng bộ khôi phục hình học/màu/gradient dành cho poster hoặc LaMa dành cho
  texture. Manifest luôn đánh dấu đây là nền tổng hợp vì pixel vốn bị che không tồn tại trong ảnh phẳng.
- Xuất bundle thật gồm PSD pixel-layer có group, OpenRaster 0.0.6, ZIP PNG/mask di động, preview,
  layer map, contact sheet, OCR JSON và manifest có hash/QA; không đổi đuôi giả và không ghi PSB giả.
- Kiểm tra ảnh tái ghép từ chính asset RGBA 8-bit đã xuất, mở lại PSD, kiểm tra thứ tự/hierarchy/offset,
  và validate cấu trúc cùng composite của ORA trước khi công bố bundle.
- Nhúng cùng profile sRGB đã chuẩn hóa vào mọi PNG màu, các PNG màu trong ORA và resource ICC của
  PSD; mask alpha `L` cố ý không mang profile RGB. Runtime mở lại và xác minh profile byte-for-byte.
- Thêm `setup_v5.ps1`, lock dependency, revision model và SHA-256 cho LaMa cùng dữ liệu OCR tiếng Việt;
  hỗ trợ CUDA cục bộ hoặc CPU chậm hơn, không tải ảnh người dùng lên cloud.

### Changed

- V2 Fast, V3 High và hai chế độ V4 được giữ nguyên; V5 có thư mục input/output, runtime, model và
  kiểm thử riêng nhưng vẫn dùng chung một lệnh `upscale`.
- Phiên bản ứng dụng tăng lên `0.3.0`; tài liệu công khai phân biệt rõ raster layer suy luận với layer
  gốc, PSD adapter với bản mở ORA/ZIP, và chi tiết nhìn thấy với phần nền vô hình chỉ có thể tổng hợp.

## [0.2.0] - 2026-08-03

### Added

- Thêm `upscale print`: V4 Print tạo hoặc tái dùng master V3 native x4 đã kiểm định,
  rồi chạy lượt HAT thứ hai đồng đều trên toàn ảnh và resample trực tiếp về đúng xN.
- Engine Deep chia tile có vùng chồng để vừa VRAM, hòa trộn cosine và không tạo/lưu ảnh x16
  trung gian; gate native-x4 chọn ứng viên đồng nhất tốt nhất sau đối chứng guarded-USM no-Deep.
- Thêm layer G'MIC/VTracer gồm các path Bezier thật nhưng độc lập và opacity `0` mặc định.
  Quyết định này giữ khả năng chỉnh sửa mà không phủ contour/posterization lên gradient, bóng
  và texture của bản in.
- Thêm `upscale vector` toàn path cho logo/artwork phẳng; giữ riêng để không posterize
  ảnh tự nhiên trong chế độ in mặc định.
- Xuất PDF 1.6 bằng Scribus với nhận dạng PDF/X-4, OutputIntent ICC, BleedBox/TrimBox và XMP
  PDF/X; gọi đúng phạm vi kiểm tra cục bộ là structural self-check, không phải chứng nhận độc lập.
- Tạo PNG xem nhanh từ chính SVG master bằng resvg; với V4 Print, `n` quyết định trực tiếp
  kích thước raster đã thắng gate hoặc fallback được nhúng và nhìn thấy.
- Cache riêng V3 native x4 và V4 Deep xN bằng SHA-256 của nguồn canonical, engine, model,
  cấu hình, master và hệ số scale; cache cũ/hỏng không còn được tin chỉ vì trùng tên.
- Hỗ trợ `--width-mm`, `--bleed-mm`, `--profile-name` và batch thư mục cho V4.
- Tự chọn tỷ lệ PDF `1:d` nhỏ nhất để cả khổ thành phẩm và bleed không vượt 5.000 mm;
  mở lại PDF và đo thật MediaBox/TrimBox/BleedBox trước khi công bố.
- Kiểm tra image/path đúng theo từng mode, đo PSNR/sai số màu/edge F1 và xác nhận layer vector
  opacity `0` không làm thay đổi raster được chọn.
- Sửa bleed thành phẩm cho PDF thu nhỏ: bleed trang được chia đúng theo mẫu số tỷ lệ.
- Thêm cổng so sánh từng crop độc lập ở đúng thang pixel V3-native, không còn seam mosaic giả:
  giữ SSIM/edge, chặn ringing/overshoot, bắt buộc tăng cả một chỉ báo edge-strength lẫn một chỉ báo
  energy/detail trên tối thiểu 75% crop. Guarded-USM là đối chứng; Deep chỉ được chọn nếu thắng có
  ý nghĩa và không giảm retention. Raster xN thật được chấm lại; không đạt thì fallback V3.
- Chuẩn hóa EXIF, ICC và alpha một lần sang sRGB/nền trắng trước mọi engine; chặn sớm kế
  hoạch vượt RAM/ổ đĩa và thêm hard cap kể cả khi dùng `--allow-huge`.
- Sửa hệ số lẻ cho ảnh không vuông và buộc test Deep chạy trong runtime CUDA thay vì bị skip.
- Sửa Scribus nhập SVG theo kích thước tự nhiên làm artwork nhỏ ở góc PDF: exporter nay scale
  đồng tỷ lệ, căn giữa, đo độ phủ TrimBox + bleed và ghi placement QA vào manifest.
- Đổi khóa canonical cache sang hash pixel sRGB ổn định, không còn miss cache vì timestamp ICC.
- Yêu cầu preflight độc lập bằng profile nhà in/Acrobat/callas/Enfocus hoặc GWG Sign & Display
  trước đơn hàng lớn.
- Mỗi bundle V4 gồm SVG chỉnh sửa, PDF/X-4 giao in và PNG xem nhanh; manifest kỹ thuật được
  cất riêng trong `APP/manifests`.

### Changed

- Lệnh gốc nay điều phối bốn chế độ V2 Fast, V3 High, V4 Print và V4 Vector.
- Tài liệu phân biệt rõ ảnh raster AI với vector reconstruction, khổ in, tỷ lệ và ICC của
  nhà in.
- Tài liệu ghi runtime tham khảo RTX 3060 12 GB, điều kiện tái sử dụng cache, giới hạn của
  auto-trace, structural self-check PDF/X-4 và yêu cầu preflight độc lập theo nhà in/GWG.

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
