# Nguồn kỹ thuật V7

V7 tách riêng việc phục dựng raster, nhận dạng chữ và đề xuất ngôn ngữ. Không
model nào được coi là nguồn có thẩm quyền để thay đổi câu chữ của khách hàng.
Super-resolution dự đoán pixel hợp lý; nó không thể chứng minh một ký tự, giá
hoặc chi tiết sản phẩm vốn không đọc được đã từng chứa nội dung gì.

## Đặc tả và dự án chính thức

- PaddleOCR 3.x và họ model PP-OCRv6:
  <https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html>
- Kho mã PaddleOCR chính thức (Apache-2.0):
  <https://github.com/PaddlePaddle/PaddleOCR>
- Tài liệu Tesseract 5 và `tessdata_best` tiếng Việt chính thức (Apache-2.0):
  <https://tesseract-ocr.github.io/tessdoc/>
- Unicode Standard Annex #15. V7 dùng NFC và cố ý không compatibility-fold bằng
  NFKC đối với nội dung khách hàng: <https://unicode.org/reports/tr15/>
- Mô hình shaping của HarfBuzz và OpenType features:
  <https://harfbuzz.github.io/shaping-concepts.html>
- OpenType 1.9.1 và quyền nhúng font trong bảng OS/2:
  <https://learn.microsoft.com/en-us/typography/opentype/spec/>
- OpenType Font Variations (`fvar`) dùng cho việc khớp weight/width có giới hạn:
  <https://learn.microsoft.com/en-us/typography/opentype/spec/fvar>
- fontTools `ttLib` để đọc cmap, name, metrics và OS/2:
  <https://fonttools.readthedocs.io/en/latest/ttLib/>
- Swin2SR chính thức và bản phát hành checkpoint (Apache-2.0):
  <https://github.com/mv-lab/swin2sr> và
  <https://github.com/mv-lab/swin2sr/releases/tag/v0.0.1>
- SwinIR chính thức (Apache-2.0), dòng kiến trúc restoration transformer được
  tham khảo cho đường fidelity cục bộ: <https://github.com/JingyunLiang/SwinIR>

## Chính sách phục dựng raster

Đường raster V7 mặc định chỉ dùng checkpoint Swin2SR real-world x4 PSNR/fidelity
cục bộ qua runner CUDA single-model của V3. Nó không dùng HAT/Real-ESRGAN GAN và
không fusion ba model. Ở `n=1`, model vẫn tạo dự đoán native x4; V7 downsample có
kiểm soát một lần về kích thước nguồn.

Trước SR là deblur luminance có giới hạn, denoise theo mức nhiễu và tăng chi tiết
quan sát được. Mỗi bước có gate và có thể rollback. Nếu backend lỗi hoặc ứng viên
SR bị loại, V7 dùng kết quả classical/Lanczos có bảo vệ và bắt buộc công bố việc
fallback trong QA cùng manifest.

Hai dự án phục dựng chính thức sau đã được nghiên cứu làm ứng viên tương lai nhưng
checkpoint của chúng **không** nằm trong runtime V7 mặc định:

- NAFNet chính thức (MIT): <https://github.com/megvii-research/NAFNet>
- Restormer chính thức (MIT): <https://github.com/swz30/Restormer>

Không được mô tả chúng là stage đang hoạt động nếu một phiên bản sau chưa pin
checkpoint, kiểm tra license/hash và vượt cùng hệ QA nhận biết nguồn.

## Nghiên cứu định hướng kiến trúc chữ

- Shimoda et al., “De-Rendering Stylized Texts”, ICCV 2021. Bài báo định hướng
  việc khôi phục tham số chữ/style/nền rồi render lại, thay vì tiếp tục sharpen
  glyph raster hỏng:
  <https://openaccess.thecvf.com/content/ICCV2021/html/Shimoda_De-Rendering_Stylized_Texts_ICCV_2021_paper.html>
- FASTER, WACV 2025, framework scene-text editing/rendering không phụ thuộc font:
  <https://openaccess.thecvf.com/content/WACV2025/html/Das_FASTER_A_Font-Agnostic_Scene_Text_Editing_and_Rendering_Framework_WACV_2025_paper.html>

Các dự án scene-text sinh ảnh chỉ là tài liệu nghiên cứu, không phải thẩm quyền
nội dung. V7 dùng mask, khớp nền và render font có giới hạn, đồng thời đưa mọi
thay đổi do model ngôn ngữ đề xuất cho người dùng duyệt.

## Nguồn gốc model đã pin

- `PP-OCRv6_medium_det` và `PP-OCRv6_medium_rec`: archive inference chính thức
  từ `paddle-model-ecology.bj.bcebos.com`; file runtime được kiểm bằng
  `MODEL_SHA256SUMS.txt`.
- `nrl-ai/vn-spell-correction-base`, Apache-2.0, revision chính xác
  `61596a71696ba360ae828f9db3806610afedf6d3`; chỉ cung cấp đề xuất.
- Tesseract Vietnamese `tessdata_best` được tải vào thư mục model riêng bị Git
  bỏ qua ở revision `e2aad9b983032bb1beff9133104a67cdbb87ca4d` và kiểm SHA-256
  trước khi được dùng như một phiếu OCR độc lập.
